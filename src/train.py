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
import pickle

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Image transformations applied on-the-fly to each image loaded
image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Converts PIL image to [C,H,W] tensor in [0,1]
        # Normalize can be added here if needed
    ]
)

# Load your dictionaries (waveform_dict and image_paths_dict)
with open(config["waveform_dict_path"], "rb") as f:
    waveform_dict = pickle.load(f)

with open(config["image_paths_dict_path"], "rb") as f:
    image_paths_dict = pickle.load(f)

# Initialize dataset with lazy image loading
dataset = ImageWaveformDataset(
    waveform_dict=waveform_dict,
    image_paths_dict=image_paths_dict,
    transform=image_transform,
)

# Use DataLoader with multiple workers and pinned memory for speed
# Tune batch_size and num_workers based on your system's CPU, RAM, and disk I/O
dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4,  # Increase if you have more CPU cores and enough RAM
    pin_memory=True,  # Speeds up transfer to CUDA GPU
)

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models and move to device
image_encoder = ImageEncoder(config["latent_dim"]).to(device)
waveform_decoder = WaveformDecoder(config["latent_dim"]).to(device)
waveform_encoder = WaveformEncoder(config["latent_dim"]).to(device)
image_decoder = ImageDecoder(config["latent_dim"]).to(device)

# Set up optimizer for all model parameters
optimizer = optim.Adam(
    list(image_encoder.parameters())
    + list(waveform_decoder.parameters())
    + list(waveform_encoder.parameters())
    + list(image_decoder.parameters()),
    lr=config["learning_rate"],
)

# Training loop
for epoch in range(config["epochs"]):
    image_encoder.train()
    waveform_decoder.train()
    waveform_encoder.train()
    image_decoder.train()

    total_loss = 0.0

    # tqdm progress bar with batch count and epoch info
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", ncols=100)
    for images, waveforms in pbar:
        # Move batch to device
        images = images.to(device, non_blocking=True)
        waveforms = waveforms.to(device, non_blocking=True)

        # Forward pass: image to waveform
        z_img = image_encoder(images)
        recon_wave = waveform_decoder(z_img)
        z_wave_from_recon = waveform_encoder(recon_wave)

        # Forward pass: waveform to image
        z_wave = waveform_encoder(waveforms)
        recon_img = image_decoder(z_wave)
        z_img_from_recon = image_encoder(recon_img)

        # Compute losses
        wave_loss = F.mse_loss(recon_wave, waveforms)
        img_loss = F.mse_loss(recon_img, images)
        latent_loss = latent_alignment_loss(z_img, z_wave)

        loss = wave_loss + img_loss + latent_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with current batch loss
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

    # Save model checkpoints periodically
    if (epoch + 1) % config["save_every"] == 0:
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch+1}.pt")
        torch.save(
            {
                "image_encoder": image_encoder.state_dict(),
                "waveform_decoder": waveform_decoder.state_dict(),
                "waveform_encoder": waveform_encoder.state_dict(),
                "image_decoder": image_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

print("Training complete.")
