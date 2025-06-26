# models/image_decoder.py
import torch
import torch.nn as nn
import torchvision.models as models


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 8 * 8 * 256), nn.ReLU())

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),  # 32→64
            nn.Sigmoid(),
        )

    def forward(self, z, skip_connections):
        x = self.fc(z).view(-1, 256, 8, 8)

        x = torch.cat([x, skip_connections[0]], dim=1)  # +256 → 512
        x = self.up1(x)

        x = torch.cat([x, skip_connections[1]], dim=1)  # +128 → 256
        x = self.up2(x)

        x = torch.cat([x, skip_connections[2]], dim=1)  # +64 → 128
        x = self.up3(x)

        return x
