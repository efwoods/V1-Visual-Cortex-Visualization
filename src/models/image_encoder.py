# models/image_encoder.py
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove classifier
        self.fc = nn.Linear(base.fc.in_features, latent_dim)

    def forward(self, x):
        x = self.features(x).squeeze()
        return self.fc(x)
