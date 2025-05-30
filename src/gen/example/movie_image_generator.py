import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

# ===== Hyperparameters =====
EPOCHS = 20
BATCH_SIZE = 64
LATENT_DIM = 100
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_LABEL = 1.
FAKE_LABEL = 0.

# ===== Create Output Folders =====
os.makedirs("output", exist_ok=True)
os.makedirs("runs", exist_ok=True)

# ===== Generator =====
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ===== Discriminator =====
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# ===== Dataset (MNIST) =====
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root="data", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ===== Initialize Models =====
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ===== TensorBoard Writer =====
writer = SummaryWriter(log_dir="runs/gan_experiment")

# ===== Training Loop =====
step = 0
for epoch in tqdm(range(EPOCHS), desc="Training", ascii="░▒▓█"):
    for i, (real_cpu, _) in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = real_cpu.to(DEVICE)
        b_size = real_cpu.size(0)

        label = torch.full((b_size,), REAL_LABEL, device=DEVICE)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        netG.zero_grad()
        label.fill_(REAL_LABEL)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch}/{EPOCHS}][{i}/{len(dataloader)}] Loss_D: {(errD_real + errD_fake):.4f} Loss_G: {errG:.4f}")
            writer.add_scalar("Loss/Discriminator", (errD_real + errD_fake).item(), step)
            writer.add_scalar("Loss/Generator", errG.item(), step)
            writer.add_images("FakeImages", fake[:16], step, dataformats="NCHW")
        step += 1

    utils.save_image(fake.detach()[:64], f"output/epoch_{epoch:03}.png", normalize=True)

writer.close()
