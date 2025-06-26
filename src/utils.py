# utils.py
import torch.nn.functional as F


def latent_alignment_loss(z_image, z_waveform):
    mse = F.mse_loss(z_image, z_waveform)
    # Assymetric alignment with cosine similarity may add structure
    cos = 1 - F.cosine_similarity(z_image, z_waveform).mean()
    return mse + 0.5 * cos
