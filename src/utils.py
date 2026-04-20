"""PSNR/SSIM metrics and plotting helpers."""

from __future__ import annotations

import os
from datetime import datetime
import matplotlib.pyplot as plt

def save_training_plots(
    train_losses,
    val_losses,
    val_psnrs,
    model_name,
    save_dir="../images",
    time_suffix=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if time_suffix is None:
        time_suffix = datetime.now().strftime("%Hh%M")

    safe_model_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name)
    ).strip("_")
    if not safe_model_name:
        safe_model_name = "model"

    suffix = f"_{safe_model_name}__{time_suffix}"

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_curve{suffix}.png'))
    plt.close()

    # Plot PSNR
    plt.figure(figsize=(10, 5))
    plt.plot(val_psnrs, label='Validation PSNR (dB)', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('Validation Peak Signal-to-Noise Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'psnr_curve{suffix}.png'))
    plt.close()
    
    print(f"Charts saved to {save_dir}/")

if __name__ == "__main__":
    # Quick test with dummy data
    train_losses = [0.1, 0.08, 0.06, 0.05]
    val_losses = [0.12, 0.09, 0.07, 0.06]
    val_psnrs = [20, 25, 30, 35]
    save_training_plots(train_losses, val_losses, val_psnrs, model_name="test_model")
