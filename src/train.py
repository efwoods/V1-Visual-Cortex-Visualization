from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models.image_encoder import ImageEncoder
from models.image_decoder import ImageDecoder
from models.waveform_decoder import WaveformDecoder
from models.waveform_encoder import WaveformEncoder
from data.dataset import ImageWaveformDataset
from utils import latent_alignment_loss
import pickle

from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

from torch.optim.lr_scheduler import LambdaLR

PERCEPTUAL_WEIGHT = 0.1


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tensor_to_image(tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img, 0, 1)
    return img


@torch.no_grad()
def evaluate_metrics(image_preds, image_targets):
    batch_size = image_preds.shape[0]
    mse_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0

    for i in range(batch_size):
        pred = tensor_to_image(image_preds[i])
        target = tensor_to_image(image_targets[i])

        mse = np.mean((pred - target) ** 2)
        psnr = compare_psnr(target, pred, data_range=1.0)
        ssim = compare_ssim(target, pred, multichannel=True, data_range=1.0)

        mse_total += mse
        psnr_total += psnr
        ssim_total += ssim

    return mse_total / batch_size, psnr_total / batch_size, ssim_total / batch_size


def main():
    config = load_config()

    log_dir = os.path.join(
        "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained VGG16 feature extractor for perceptual loss
    vgg = vgg16(weights="IMAGENET1K_V1").features.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False  # freeze weights

    # Create extractor for relu2_2 layer (layer index 16)
    feature_extractor = create_feature_extractor(vgg, return_nodes={"16": "features"})

    def perceptual_loss(img1, img2):
        # Input: shape (B, 3, H, W), range [0,1]
        return torch.nn.functional.mse_loss(
            feature_extractor(img1)["features"], feature_extractor(img2)["features"]
        )

    # Data transforms
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load dataset dicts
    with open(config["waveform_dict_path"], "rb") as f:
        waveform_dict = pickle.load(f)
    with open(config["image_paths_dict_path"], "rb") as f:
        image_paths_dict = pickle.load(f)

    dataset = ImageWaveformDataset(waveform_dict, image_paths_dict, transform=transform)

    # Split dataset into train/test
    test_ratio = 0.1
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Save test metadata
    test_metadata = {
        "indices": test_dataset.indices,
        "timestamp": datetime.datetime.now().isoformat(),
        "split_ratio": test_ratio,
        "total_samples": len(dataset),
    }
    with open("test_dataset_meta.pkl", "wb") as f:
        pickle.dump(test_metadata, f)

    # Further split train into train/val
    val_ratio = 0.2
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize models
    image_encoder = ImageEncoder(latent_dim=config["latent_dim"]).to(device)
    image_decoder = ImageDecoder(latent_dim=config["latent_dim"]).to(device)
    waveform_decoder = WaveformDecoder(latent_dim=config["latent_dim"]).to(device)
    waveform_encoder = WaveformEncoder(latent_dim=config["latent_dim"]).to(device)

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(
        list(image_encoder.parameters())
        + list(image_decoder.parameters())
        + list(waveform_encoder.parameters())
        + list(waveform_decoder.parameters()),
        lr=float(config["learning_rate"]),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / 10))

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    best_val_mse = float("inf")
    patience = 10
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, config["epochs"] + 1):
        # ---------- Training ----------
        image_encoder.train()
        image_decoder.train()
        waveform_encoder.train()
        waveform_decoder.train()

        train_loss = 0
        for images, waveforms in tqdm(
            train_loader, desc=f"[Train] Epoch {epoch}/{config['epochs']}"
        ):
            images = images.to(device)
            waveforms = waveforms.to(device)

            # Forward pass
            z_img, _ = image_encoder(images)
            recon_waveform = waveform_decoder(z_img)
            z_waveform_from_img = waveform_encoder(recon_waveform)

            z_waveform = waveform_encoder(waveforms)
            _, skip_feats = image_encoder(images)  # get skip features here
            recon_image = image_decoder(
                z_waveform, skip_feats
            )  # <-- FIX: pass skip_feats

            # Losses
            loss_align = latent_alignment_loss(z_img, z_waveform_from_img)
            # Upsample recon_image from 64x64 to 224x224 to match ground truth and VGG
            recon_image = torch.nn.functional.interpolate(
                recon_image, size=(224, 224), mode="bilinear", align_corners=False
            )

            loss_recon_image = torch.nn.functional.mse_loss(recon_image, images)
            loss_perceptual = perceptual_loss(recon_image, images)

            loss = loss_align + loss_recon_image + PERCEPTUAL_WEIGHT * loss_perceptual

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters())
                + list(image_decoder.parameters())
                + list(waveform_encoder.parameters())
                + list(waveform_decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # ---------- Validation ----------
        image_encoder.eval()
        image_decoder.eval()
        waveform_encoder.eval()
        waveform_decoder.eval()

        val_loss = 0
        total_batches = 0

        with torch.no_grad():
            for images, waveforms in tqdm(val_loader, desc="[Val] Evaluating"):
                images = images.to(device)
                waveforms = waveforms.to(device)

                z_waveform = waveform_encoder(waveforms)
                _, skip_feats = image_encoder(images)
                recon_image = image_decoder(z_waveform, skip_feats)
                recon_image = torch.nn.functional.interpolate(
                    recon_image, size=(224, 224), mode="bilinear", align_corners=False
                )

                loss = torch.nn.functional.mse_loss(recon_image, images)
                val_loss += loss.item()

                total_batches += 1

        avg_val_loss = val_loss / len(val_loader)

        # TensorBoard Logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if epoch % 5 == 0:
            gt_img = tensor_to_image(images[0])
            rec_img = tensor_to_image(recon_image[0])
            gt_tensor = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0)
            rec_tensor = torch.tensor(rec_img).permute(2, 0, 1).unsqueeze(0)
            writer.add_images(
                "GroundTruth vs Reconstruction",
                torch.cat([gt_tensor, rec_tensor], dim=0),
                epoch,
            )

        # Early stopping logic
        if avg_val_loss < best_val_mse:
            best_val_mse = avg_val_loss
            epochs_no_improve = 0
            torch.save({...}, os.path.join(config["checkpoint_dir"], "best_model.pt"))
            print(f"[Checkpoint] Saved best model at epoch {epoch}")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStopping] No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"[EarlyStopping] Stopping early at epoch {epoch}")
                early_stop = True
                break

        print(
            f"\n[Epoch {epoch}] "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # Save best models (simulation and relay parts)
        if avg_val_loss < best_val_mse:
            best_val_mse = avg_val_loss

            torch.save(
                {
                    "image_encoder": image_encoder.state_dict(),
                    "waveform_decoder": waveform_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(config["checkpoint_dir"], "simulation_model.pt"),
            )

            torch.save(
                {
                    "waveform_encoder": waveform_encoder.state_dict(),
                    "image_decoder": image_decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(config["checkpoint_dir"], "relay_model.pt"),
            )
            print(
                f"[Checkpoint] Best simulation_model.pt and relay_model.pt saved at epoch {epoch} with MSE {avg_val_loss:.6f}"
            )

        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
