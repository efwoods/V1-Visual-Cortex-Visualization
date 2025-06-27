# train.py
import os
import datetime
import pickle
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from models.image_encoder import ImageEncoder
from models.image_decoder import ImageDecoder
from models.waveform_decoder import WaveformDecoder
from models.waveform_encoder import WaveformEncoder
from data.dataset import ImageWaveformDataset
from utils import latent_alignment_loss
import torchvision.utils as vutils


# Constants
PERCEPTUAL_WEIGHT = 0.1
USE_PERCEPTUAL_LOSS = False  # Set True to enable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_FROM_CHECKPOINT = True


def load_all_from_checkpoint(
    image_enc, wave_enc, image_dec, wave_dec, optimizer, config, path=None
):
    if path is None:
        path = os.path.join(
            config["checkpoint_dir"], "checkpoint.pt"
        )  # Assuming checkpoint
    checkpoint = torch.load(path)
    image_enc.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "image_encoder.pt"))
    )
    wave_enc.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "waveform_encoder.pt"))
    )
    image_dec.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "image_decoder.pt"))
    )
    wave_dec.load_state_dict(
        torch.load(os.path.join(config["checkpoint_dir"], "waveform_decoder.pt"))
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val"],
        checkpoint.get("logdir", None),
    )


def save_all_to_checkpoint(
    image_enc,
    wave_enc,
    image_dec,
    wave_dec,
    optimizer,
    epoch,
    best_val,
    config,
    logdir,
    path=None,
):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    if path is None:
        path = os.path.join(config["checkpoint_dir"], "checkpoint.pt")
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "logdir": config.get("logdir", logdir),  # Add this line
    }
    torch.save(checkpoint, path)

    torch.save(
        image_enc.state_dict(),
        os.path.join(config["checkpoint_dir"], "image_encoder.pt"),
    )
    torch.save(
        wave_enc.state_dict(),
        os.path.join(config["checkpoint_dir"], "waveform_encoder.pt"),
    )
    torch.save(
        image_dec.state_dict(),
        os.path.join(config["checkpoint_dir"], "image_decoder.pt"),
    )
    torch.save(
        wave_dec.state_dict(),
        os.path.join(config["checkpoint_dir"], "waveform_decoder.pt"),
    )

    print(f"Saved best_model at epoch {epoch}")
    print(f"Best validation loss: {best_val}")


# Perceptual loss setup
def perceptual_loss(img1, img2, extractor):
    """
    Computes MSE between the features extracted with VGG between the reconstructed and original image.
    """
    feats1 = extractor(img1)["features"]
    feats2 = extractor(img2)["features"]
    return torch.nn.functional.mse_loss(feats1, feats2)


@torch.no_grad()
def evaluate_metrics(preds, targets):
    mse, psnr, ssim = 0.0, 0.0, 0.0
    for p, t in zip(preds, targets):
        p_img = p.detach().cpu().numpy().transpose(1, 2, 0)
        t_img = t.detach().cpu().numpy().transpose(1, 2, 0)
        mse += np.mean((p_img - t_img) ** 2)
        psnr += compare_psnr(t_img, p_img, data_range=1.0)
        ssim += compare_ssim(t_img, p_img, multichannel=True, data_range=1.0)
    n = len(preds)
    return mse / n, psnr / n, ssim / n


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def log_side_by_side_images(
    writer, original_imgs, reconstructed_imgs, epoch, tag="Images"
):
    """
    Logs a grid of side-by-side original and reconstructed images to TensorBoard.

    original_imgs, reconstructed_imgs: tensors with shape (B, C, H, W) in [0, 1]
    """
    # Clamp to [0, 1] if needed
    original_imgs = torch.clamp(original_imgs, 0, 1)
    reconstructed_imgs = torch.clamp(reconstructed_imgs, 0, 1)

    # Concatenate images side-by-side for each sample along width (dim=3)
    side_by_side = torch.cat(
        [original_imgs, reconstructed_imgs], dim=3
    )  # (B, C, H, 2*W)

    # Make a grid (e.g. max 8 images per grid)
    grid = vutils.make_grid(side_by_side, nrow=8, padding=2)

    # Write to TensorBoard
    writer.add_image(tag, grid, epoch)


def main(logdir=None):

    config = load_config()

    # TensorBoard
    # Resume logdir if specified, else create new
    os.makedirs("runs", exist_ok=True)
    if logdir is None:
        logdir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # VGG for perceptual loss
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    extractor = create_feature_extractor(vgg, return_nodes={"16": "features"})

    # Data
    if USE_PERCEPTUAL_LOSS:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
    else:
        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
    with open(config["waveform_dict_path"], "rb") as f:
        waveform_dict = pickle.load(f)
    with open(config["image_paths_dict_path"], "rb") as f:
        image_paths = pickle.load(f)
    dataset = ImageWaveformDataset(waveform_dict, image_paths, transform)

    # Splits
    test_ratio = config.get("test_split", 0.1)
    test_size = int(len(dataset) * test_ratio)
    trainval_size = len(dataset) - test_size
    trainval_set, test_set = random_split(dataset, [trainval_size, test_size])
    # Save test meta
    meta = {
        "indices": test_set.indices,
        "timestamp": datetime.datetime.now().isoformat(),
        "total": len(dataset),
    }
    with open("test_dataset_metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    val_ratio = config.get("val_split", 0.2)
    val_size = int(len(trainval_set) * val_ratio)
    train_size = len(trainval_set) - val_size
    train_set, val_set = random_split(trainval_set, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
    )

    # Models
    image_enc = ImageEncoder(latent_dim=config["latent_dim"]).to(DEVICE)
    image_dec = ImageDecoder(latent_dim=config["latent_dim"]).to(DEVICE)
    wave_dec = WaveformDecoder(latent_dim=config["latent_dim"]).to(DEVICE)
    wave_enc = WaveformEncoder(latent_dim=config["latent_dim"]).to(DEVICE)

    if torch.__version__ >= "2.0.0":
        image_enc = torch.compile(image_enc)
        image_dec = torch.compile(image_dec)
        wave_enc = torch.compile(wave_enc)
        wave_dec = torch.compile(wave_dec)

    params = (
        list(image_enc.parameters())
        + list(image_dec.parameters())
        + list(wave_enc.parameters())
        + list(wave_dec.parameters())
    )

    original_batch_size = int(config["original_batch_size"])  # your baseline batch size
    current_batch_size = int(config["batch_size"])  # new batch size

    base_lr = float(config["learning_rate"])
    scaled_lr = base_lr * (current_batch_size / original_batch_size)
    print(
        f"Scaling learning rate from {base_lr} to {scaled_lr} due to batch size change"
    )

    optimizer = torch.optim.Adam(params, lr=scaled_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, e / 10))

    best_val = float("inf")
    epochs_no_improve = 0
    start_epoch = 1
    if LOAD_FROM_CHECKPOINT:
        start_epoch, best_val, logdir = load_all_from_checkpoint(
            image_enc, wave_enc, image_dec, wave_dec, optimizer, config=config
        )
    for epoch in range(start_epoch, config["epochs"] + 1):
        # Train
        image_enc.train()
        image_dec.train()
        wave_enc.train()
        wave_dec.train()
        total_loss = 0.0
        for imgs, waves in tqdm(train_loader, desc=f"Train {epoch}/{config['epochs']}"):
            imgs = imgs.to(DEVICE)
            waves = waves.to(DEVICE)
            # Encode image -> latent + skips
            image_latent, skips = image_enc(imgs)  # Used for latent alignment loss
            # Synthetic waveform -> latent
            synth_wave = wave_dec(image_latent)
            synthetic_waveform_latent = wave_enc(
                synth_wave
            )  # Used for latent alignment loss between the image encoder and waveform encoder (both encodings encode into the same space)

            # Real waveform -> latent
            real_waveform_latent = wave_enc(waves)
            # Decode real latent -> reconstructed image
            reconstructed_image = image_dec(real_waveform_latent, skips)
            if USE_PERCEPTUAL_LOSS:
                reconstructed_image_upscaled_for_vgg_quality = (
                    torch.nn.functional.interpolate(
                        reconstructed_image,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                )

            # Losses
            loss_align = latent_alignment_loss(
                image_latent, synthetic_waveform_latent
            )  # error between the encoded image and encoded waveform

            if USE_PERCEPTUAL_LOSS:
                loss_mse = torch.nn.functional.mse_loss(
                    reconstructed_image_upscaled_for_vgg_quality, imgs
                )  # error between reconstructed and original image

                loss_perceptual = perceptual_loss(
                    reconstructed_image_upscaled_for_vgg_quality, imgs, extractor
                )  # Quality error between the reconstructed image and the original image
            else:
                loss_mse = torch.nn.functional.mse_loss(
                    reconstructed_image, imgs
                )  # error between reconstructed and original image
                loss_perceptual = 0.0
            loss = loss_align + loss_mse + PERCEPTUAL_WEIGHT * loss_perceptual

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # Validate
        image_enc.eval()
        image_dec.eval()
        wave_enc.eval()
        wave_dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, waves in tqdm(val_loader, desc="Validate"):
                imgs = imgs.to(DEVICE)
                waves = waves.to(DEVICE)
                image_latent, skips = image_enc(imgs)
                real_waveform_latent = wave_enc(waves)
                reconstructed_image = image_dec(real_waveform_latent, skips)
                if USE_PERCEPTUAL_LOSS:
                    reconstructed_image_upscaled_for_vgg_quality = (
                        torch.nn.functional.interpolate(
                            reconstructed_image,
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                    val_loss += torch.nn.functional.mse_loss(
                        reconstructed_image_upscaled_for_vgg_quality, imgs
                    ).item()
                    # Log side-by-side images (original vs reconstructed upscaled)
                    log_side_by_side_images(
                        writer,
                        imgs,
                        reconstructed_image_upscaled_for_vgg_quality,
                        epoch,
                        tag="Val/Original_vs_Reconstructed",
                    )
                else:
                    val_loss += torch.nn.functional.mse_loss(
                        reconstructed_image, imgs
                    ).item()
                    # Log side-by-side images (original vs reconstructed upscaled)
                    log_side_by_side_images(
                        writer,
                        imgs,
                        reconstructed_image,
                        epoch,
                        tag="Val/Original_vs_Reconstructed",
                    )
        avg_val = val_loss / len(val_loader)

        # Logging
        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", avg_val, epoch)
        writer.add_scalar("Loss/loss_mse (latent alignment)", loss_mse, epoch)
        scheduler.step()

        print(f"Epoch {epoch}: Train={avg_train:.4f}, Val={avg_val:.4f}")

        # Checkpoint

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0

            save_all_to_checkpoint(
                image_enc=image_enc,
                wave_enc=wave_enc,
                image_dec=image_dec,
                wave_dec=wave_dec,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val,
                config=config,
                logdir=logdir,
            )

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.get("patience", 10):
                print(f"Early stopping at epoch {epoch}")
                break

        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
