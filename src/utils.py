# utils.py
import torch.nn.functional as F


def latent_alignment_loss(image_latent, waveform_latent):
    mse = F.mse_loss(image_latent, waveform_latent)
    # Assymetric alignment with cosine similarity may add structure
    cos = 1 - F.cosine_similarity(image_latent, waveform_latent).mean()
    return mse + 0.5 * cos
