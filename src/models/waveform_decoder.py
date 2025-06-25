# models/waveform_decoder.py
import torch.nn as nn


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_shape=(16, 48)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_shape[0] * out_shape[1]),
        )
        self.out_shape = out_shape

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, *self.out_shape)
