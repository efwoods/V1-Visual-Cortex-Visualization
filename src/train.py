# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ImageWaveformDataset
from models.image_encoder import ImageEncoder
from models.waveform_decoder import WaveformDecoder
from models.waveform_encoder import WaveformEncoder
from models.image_decoder import ImageDecoder
from utils import latent_alignment_loss
import torch.nn.functional as F
import torch.optim as optim
import yaml
import os
from tqdm import tqdm

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Prepare transforms and dataset
image_transform = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

# These should be preloaded numpy dicts
waveform_dict = torch.load(config["waveform_dict_path"])
image_paths_dict = torch.load(config["image_paths_dict_path"])

dataset = ImageWaveformDataset(
    waveform_dict, image_paths_dict, transform=image_transform
)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder = ImageEncoder(config["latent_dim"]).to(device)
waveform_decoder = WaveformDecoder(config["latent_dim"]).to(device)
waveform_encoder = WaveformEncoder(config["latent_dim"]).to(device)
image_decoder = ImageDecoder(config["latent_dim"]).to(device)

# Optimizers
params = (
    list(image_encoder.parameters())
    + list(waveform_decoder.parameters())
    + list(waveform_encoder.parameters())
    + list(image_decoder.parameters())
)
optimizer = optim.Adam(params, lr=config["learning_rate"])

# Training Loop
for epoch in range(config["epochs"]):
    image_encoder.train()
    waveform_decoder.train()
    waveform_encoder.train()
    image_decoder.train()

    total_loss = 0.0
    for images, waveforms in tqdm(dataloader):
        images = images.to(device)
        waveforms = waveforms.to(device)

        # Forward simulation path
        z_img = image_encoder(images)
        recon_wave = waveform_decoder(z_img)
        z_wave_from_recon = waveform_encoder(recon_wave)

        # Forward reconstruction path
        z_wave = waveform_encoder(waveforms)
        recon_img = image_decoder(z_wave)
        z_img_from_recon = image_encoder(recon_img)

        # Losses
        wave_loss = F.mse_loss(recon_wave, waveforms)
        img_loss = F.mse_loss(recon_img, images)
        latent_loss = latent_alignment_loss(z_img, z_wave)

        loss = wave_loss + img_loss + latent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % config["save_every"] == 0:
        ckpt_dir = config["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "image_encoder": image_encoder.state_dict(),
                "waveform_decoder": waveform_decoder.state_dict(),
                "waveform_encoder": waveform_encoder.state_dict(),
                "image_decoder": image_decoder.state_dict(),
            },
            os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"),
        )
