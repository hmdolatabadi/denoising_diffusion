import os
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.model import UNet
import lib.dataset as dataset
from lib.diffusion import GaussianDiffusion, make_beta_schedule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape):
    samples = diffusion.p_sample_loop(model=model,
                                      shape=shape,
                                      noise_fn=torch.randn)
    return {
      'samples': (samples + 1)/2
    }


def progressive_samples_fn(model, diffusion, shape, device, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq
    )
    return {'samples': (samples + 1)/2, 'progressive_samples': (progressive_samples + 1)/2}


def bpd_fn(model, diffusion, x):
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(model=model, x_0=x, clip_denoised=True)

    return {
      'total_bpd': total_bpd_b,
      'terms_bpd': terms_bpd_bt,
      'prior_bpd': prior_bpd_b,
      'mse': mse_bt
    }


def validate(val_loader, model, diffusion):
    model.eval()
    bpd = []
    mse = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(val_loader)):
            x       = x
            metrics = bpd_fn(model, diffusion, x)

            bpd.append(metrics['total_bpd'].view(-1, 1))
            mse.append(metrics['mse'].view(-1, 1))

        bpd = torch.cat(bpd, dim=0).mean()
        mse = torch.cat(mse, dim=0).mean()

    return bpd, mse


class DDP(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf  = conf
        self.save_hyperparameters()

        self.model = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          )

        self.ema   = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          )

        self.betas = make_beta_schedule(schedule=self.conf.model.schedule.type,
                                        start=self.conf.model.schedule.beta_start,
                                        end=self.conf.model.schedule.beta_end,
                                        n_timestep=self.conf.model.schedule.n_timestep)

        self.diffusion = GaussianDiffusion(betas=self.betas,
                                           model_mean_type=self.conf.model.mean_type,
                                           model_var_type=self.conf.model.var_type,
                                           loss_type=self.conf.model.loss_type)



    def setup(self, stage):

        self.train_set, self.valid_set = dataset.get_train_data(self.conf)

    def forward(self, x):

        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):

        if self.conf.training.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.training.optimizer.lr)
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_nb):

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss   = self.diffusion.training_losses(self.model, img, time).mean()

        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.conf.training.dataloader.batch_size,
                                  shuffle=True,
                                  num_workers=self.conf.training.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.conf.training.dataloader.drop_last)

        return train_loader

    def validation_step(self, batch, batch_nb):

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss   = self.diffusion.training_losses(self.ema, img, time).mean()

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss         = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        shape  = (16, 3, self.conf.dataset.resolution, self.conf.dataset.resolution)
        sample = progressive_samples_fn(self.ema, self.diffusion, shape, device='cuda' if self.on_gpu else 'cpu')

        grid = make_grid(sample['samples'], nrow=4)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        grid = make_grid(sample['progressive_samples'].reshape(-1, 3, self.conf.dataset.resolution, self.conf.dataset.resolution), nrow=20)
        self.logger.experiment.add_image(f'progressive_generated_images', grid, self.current_epoch)
        
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.conf.validation.dataloader.batch_size,
                                  shuffle=False,
                                  num_workers=self.conf.validation.dataloader.num_workers,
                                  pin_memory=True,
                                  drop_last=self.conf.validation.dataloader.drop_last)

        return valid_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False, help="Training or evaluation?")
    parser.add_argument("--config", type=str, required=True, help="Path to config.")

    # Training specific args
    parser.add_argument("--ckpt_dir", type=str, default='ckpts', help="Path to folder to save checkpoints.")
    parser.add_argument("--ckpt_freq", type=int, default=20, help="Frequency of saving the model (in epoch).")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of available GPUs.")

    # Eval specific args
    parser.add_argument("--model_dir", type=str, default='final/cifar10.ckpt', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--prog_sample_freq", type=int, default=200, help="Progressive sample frequency.")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of generated samples in evaluation.")

    args = parser.parse_args()

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)
    denoising_diffusion_model = DDP(conf)

    if args.train:
        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.ckpt_dir, 'ddp_{epoch:02d}-{val_loss:.2f}'),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=-1,
                                              save_weights_only=True,
                                              mode='auto',
                                              period=args.ckpt_freq,
                                              prefix='')

        trainer = pl.Trainer(fast_dev_run=False,
                             gpus=args.n_gpu,
                             max_steps=conf.training.n_iter,
                             precision=conf.model.precision,
                             gradient_clip_val=1.,
                             progress_bar_refresh_rate=20,
                             checkpoint_callback=checkpoint_callback)

        trainer.fit(denoising_diffusion_model)

    else:
        
        denoising_diffusion_model.cuda()
        state_dict = torch.load(args.model_dir)
        denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
        denoising_diffusion_model.eval()

        sample = progressive_samples_fn(denoising_diffusion_model.ema,
                                        denoising_diffusion_model.diffusion,
                                        (args.n_samples, 3, conf.dataset.resolution, conf.dataset.resolution),
                                        device='cuda',
                                        include_x0_pred_freq=args.prog_sample_freq)

        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)

        for i in range(args.n_samples):

            img = sample['samples'][i]
            plt.imsave(os.path.join(args.sample_dir, f'sample_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

            img = sample['progressive_samples'][i]
            img = make_grid(img, nrow=args.prog_sample_freq)
            plt.imsave(os.path.join(args.sample_dir, f'prog_sample_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

