# Imports 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from tqdm import tqdm

# Configuration
DATA_DIR = "/home/linux-pc/gh/V1-Visual-Cortex-Visualization/data/crcns-pvc1/crcns-ringach-data/movie_frames/movie000_000.images/"
BATCH_SIZE = 128
IMAGE_SIZE = 64
EPOCHS = 100
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Loader
class MovieFramesDataset(Dataset):
    def __init__(self, root_dir, tranform=None):
        self.image_paths = sorted(glob(os.path.join(root_dir, "*.jpeg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('L') # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


dataset = MovieFramesDataset(DATA_DIR, tranform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()            
        )
    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)



# Initialize
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Labels
real_label = 1
fake_label = 0

# Training Loop
for epoch in tqdm(range(EPOCHS), desc=f"Training...", ascii="░▒▓█"):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator with real data
        netD.zero_grad()
        real_cpu = data.to(DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=DEVICE)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()

        # Generate fake image
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        label.fill_(real_label) # Generator wants D to believe its output is real
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch}/{EPOCHS}][{i}/{len(dataloader)}] Loss_D: {errD_real+errD_fake:.4f} Loss_G: {errG:.4f}")
    utils.save_image(fake.detach(), f"output/epoch_{epoch:03}.png",)