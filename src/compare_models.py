import os
from datetime import datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import NIHDenoisingDataset
from models.unet import UNet
from models.attention_unet import AttentionUNet

# Configurations
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "images"
CHECKPOINT_DIR = REPO_ROOT / "models_checkpoints"
BASELINE_CHECKPOINT_NAME = None
ATTENTION_CHECKPOINT_NAME = None
OUTPUT_DIR = REPO_ROOT / "images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_model_name(model_name: str) -> str:
    safe_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name)
    ).strip("_")
    return safe_name or "model"


def parse_model_name_from_checkpoint(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if "__" in stem:
        stem = stem.rsplit("__", 1)[0]
    return sanitize_model_name(stem)


def resolve_checkpoint(explicit_name: str | None, pattern: str) -> Path:
    if explicit_name:
        checkpoint_path = CHECKPOINT_DIR / explicit_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidates = sorted(CHECKPOINT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' found in {CHECKPOINT_DIR}")
    return candidates[-1]


def save_model_comparison_grid(noisy, base_denoised, att_denoised, clean, num_images=4, save_path="model_comparison.png"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    
    for i in range(num_images):
        # Convert tensors to numpy for plotting
        n_img = noisy[i].cpu().squeeze().numpy()
        b_img = base_denoised[i].cpu().squeeze().numpy()
        a_img = att_denoised[i].cpu().squeeze().numpy()
        c_img = clean[i].cpu().squeeze().numpy()

        # 1. Noisy Input
        axes[i, 0].imshow(n_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Noisy Input")
        axes[i, 0].axis('off')

        # 2. Baseline U-Net
        axes[i, 1].imshow(b_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Baseline U-Net")
        axes[i, 1].axis('off')

        # 3. Attention U-Net
        axes[i, 2].imshow(a_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Attention U-Net")
        axes[i, 2].axis('off')

        # 4. Ground Truth
        axes[i, 3].imshow(c_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title("Ground Truth (Clean)")
        axes[i, 3].axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / save_path
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison grid saved to {output_path}")

def main():
    print(f"Loading models on {DEVICE}...")

    baseline_checkpoint = resolve_checkpoint(BASELINE_CHECKPOINT_NAME, "unet*.pth")
    attention_checkpoint = resolve_checkpoint(ATTENTION_CHECKPOINT_NAME, "attention_unet*.pth")
    baseline_model_name = parse_model_name_from_checkpoint(baseline_checkpoint)
    attention_model_name = parse_model_name_from_checkpoint(attention_checkpoint)
    run_time_suffix = datetime.now().strftime("%Hh%M")
    
    # Load Baseline
    model_base = UNet().to(DEVICE)
    model_base.load_state_dict(torch.load(baseline_checkpoint, map_location=DEVICE, weights_only=True))
    model_base.eval()

    # Load Attention U-Net
    model_att = AttentionUNet().to(DEVICE)
    model_att.load_state_dict(torch.load(attention_checkpoint, map_location=DEVICE, weights_only=True))
    model_att.eval()

    # Recreate the dataset splits to get the exact test set
    full_dataset = NIHDenoisingDataset(str(DATA_DIR), image_size=256)
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

    print("Running inference on both models...")
    with torch.no_grad():
        base_outputs = model_base(noisy_imgs)
        att_outputs = model_att(noisy_imgs)

    save_name = f"model_comparison_{baseline_model_name}_vs_{attention_model_name}__{run_time_suffix}.png"
    save_model_comparison_grid(
        noisy_imgs,
        base_outputs,
        att_outputs,
        clean_imgs,
        num_images=4,
        save_path=save_name,
    )

if __name__ == "__main__":
    main()