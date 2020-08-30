# Denosing Diffusion Generative Models

This is an unofficial implementation of ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) in [PyTorch(Lightning)](https://github.com/PyTorchLightning/pytorch-lightning).

<p align="center">
  <img width="522" height="132" src="/misc/DDP.gif">
</p>
<p align="center">
  <img width="680" height="342" src="/misc/cifar10.png">
</p>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training Diffusion Models

To train a diffusion model, first specify model architecture and hyperparameters in `config.json`. Once specified, run this command:

```train
python diffusion_lightning.py --train --config config.json --ckpt_dir PATH_TO_CHECKPOINTS --ckpt_freq CHECKPOINT_FREQ --n_gpu NUM_AVAIL_GPUS
```

## Sample Generation

To generate samples from a trained diffusion model specified by `config.json`, run this command:

```eval
python diffusion_lightning.py --config config.json --model_dir MODEL_DIRECTORY --sample_dir PATH_TO_SAVE_SAMPLES --n_samples NUM_SAMPLES
```

## Pre-trained Models

Pre-trained diffusion models on CelebA and CIFAR-10 can be found [here](https://drive.google.com/drive/folders/1R8xmSjLSM1vXWpvZrv3mPE52ImCjVzxK?usp=sharing).

## CIFAR-10 FID Score

In the paper, the authors perform model selection using the FID score. Here, however, the model is only trained until 1000000 iterations and no model selection is performed due to limited computational resources. This way, we got an FID score of 5.1037.

## Acknowledgement

This repository is built upon the [official repository of diffusion models in TensorFlow](https://github.com/hojonathanho/diffusion) as well as parts of [this unofficial PyTorch implementation](https://github.com/rosinality/denoising-diffusion-pytorch).
