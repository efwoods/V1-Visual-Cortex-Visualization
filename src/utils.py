# utils.py
import torch.nn.functional as F


def latent_alignment_loss(z_image, z_waveform):
    return F.mse_loss(z_image, z_waveform)
