# models/image_decoder.py
import torch.nn as nn


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 8 * 8 * 64), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 8, 8)
        return self.deconv(x)
