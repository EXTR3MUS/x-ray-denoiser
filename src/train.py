"""Training loop entrypoint with AMP support."""

from __future__ import annotations

import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio

from dataset import NIHDenoisingDataset
from models.unet import UNet
from models.attention_unet import AttentionUNet
from utils import save_training_plots

# Configurations
DATA_DIR = "./data/raw/images"
BATCH_SIZE = 32
EPOCHS = 15 # Kept low for time constraints, increase if you have spare time
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # model_name = "unet"
    model_name = "attention_unet"
    time_suffix = datetime.now().strftime("%Hh%M")

    # 1. Prepare Dataset and Splits
    full_dataset = NIHDenoisingDataset(DATA_DIR, image_size=256)
    total_size = len(full_dataset)
    
    train_size = int(0.66 * total_size) # ~1000
    val_size = int(0.17 * total_size)   # ~250
    test_size = total_size - train_size - val_size # ~250
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Fixed seed for reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model, Loss, and Optimizer
    # model = UNet().to(DEVICE)
    model = AttentionUNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrics and AMP Scaler
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    train_losses, val_losses, val_psnrs = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for noisy_imgs, clean_imgs in loop:
            noisy_imgs, clean_imgs = noisy_imgs.to(DEVICE), clean_imgs.to(DEVICE)

            optimizer.zero_grad()
            
            # AMP Forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)

            # AMP Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        psnr_score = 0.0
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                noisy_imgs, clean_imgs = noisy_imgs.to(DEVICE), clean_imgs.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, clean_imgs)
                
                val_loss += loss.item()
                psnr_score += psnr_metric(outputs, clean_imgs).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_score / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_psnr)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB\n")

    # 5. Save Artifacts
    os.makedirs("./models_checkpoints", exist_ok=True)
    checkpoint_path = os.path.join(
        "./models_checkpoints", f"{model_name}__{time_suffix}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    save_training_plots(
        train_losses,
        val_losses,
        val_psnrs,
        model_name=model_name,
        time_suffix=time_suffix,
    )
    print("Training complete. Model and charts saved.")

if __name__ == "__main__":
    main()