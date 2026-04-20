import os
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import NIHDenoisingDataset
from models.attention_unet import AttentionUNet
from models.unet import UNet

# Configurations
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "images"
CHECKPOINT_DIR = REPO_ROOT / "models_checkpoints"
# Set to a specific filename in models_checkpoints, or keep None to auto-pick latest attention_unet checkpoint.
CHECKPOINT_NAME = None
OUTPUT_DIR = REPO_ROOT / "images"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


def sanitize_model_name(model_name: str) -> str:
    safe_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name)
    ).strip("_")
    return safe_name or "model"


def get_model_name_from_checkpoint(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if "__" in stem:
        model_name, _ = stem.rsplit("__", 1)
        return sanitize_model_name(model_name)

    return sanitize_model_name(stem)


def resolve_checkpoint_path() -> Path:
    if CHECKPOINT_NAME:
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidates = sorted(CHECKPOINT_DIR.glob("attention_unet*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No attention_unet checkpoints found in {CHECKPOINT_DIR}. "
            "Set CHECKPOINT_NAME to a valid file or run training first."
        )
    return candidates[-1]

def save_comparison_grid(noisy, denoised, clean, model_name, time_suffix, num_images=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    
    for i in range(num_images):
        # Convert tensors to numpy for plotting
        n_img = noisy[i].cpu().squeeze().numpy()
        d_img = denoised[i].cpu().squeeze().numpy()
        c_img = clean[i].cpu().squeeze().numpy()

        axes[i, 0].imshow(n_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Noisy Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(d_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Denoised Output")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(c_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Ground Truth (Clean)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"inference_results_{model_name}__{time_suffix}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Inference results saved to {output_path}")

def main():
    print(f"Loading model on {DEVICE}...")
    # model = UNet().to(DEVICE)
    model = AttentionUNet().to(DEVICE)
    checkpoint_path = resolve_checkpoint_path()
    model_name = get_model_name_from_checkpoint(checkpoint_path)
    time_suffix = datetime.now().strftime("%Hh%M")
    print(f"Using checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Recreate the dataset splits to get the exact test set
    full_dataset = NIHDenoisingDataset(DATA_DIR, image_size=256)
    total_size = len(full_dataset)
    train_size = int(0.66 * total_size)
    val_size = int(0.17 * total_size)
    test_size = total_size - train_size - val_size
    
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Load a single batch for visualization
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    noisy_imgs, clean_imgs = next(iter(test_loader))
    noisy_imgs = noisy_imgs.to(DEVICE)

    print("Running inference...")
    with torch.no_grad():
        denoised_imgs = model(noisy_imgs)

    save_comparison_grid(
        noisy_imgs,
        denoised_imgs,
        clean_imgs,
        model_name=model_name,
        time_suffix=time_suffix,
        num_images=4,
    )

if __name__ == "__main__":
    main()