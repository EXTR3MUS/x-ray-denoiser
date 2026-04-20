"""Dataloader and synthetic noise injection utilities."""

from __future__ import annotations

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NIHDenoisingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, noise_factor=0.05):
        """
        Args:
            data_dir (str): Path to the folder containing the extracted NIH .png images.
            image_size (int): Target resolution (Width and Height).
            noise_factor (float): Intensity of the synthetic Gaussian noise.
        """
        self.image_paths = glob.glob(os.path.join(data_dir, '*.png'))
        self.noise_factor = noise_factor
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # Resizes the image to the specified dimensions (image_size x image_size)
            transforms.ToTensor() # Converts PIL Image to Tensor and scales pixel values to [0.0, 1.0]  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        clean_image = Image.open(img_path).convert("L")
        clean_image = self.transform(clean_image)
        
        # Generate synthetic Gaussian noise
        noise = torch.randn_like(clean_image) * self.noise_factor
        
        noisy_image = clean_image + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return noisy_image, clean_image

if __name__ == "__main__":
    test_dir = "../data/raw/images"
    
    if os.path.exists(test_dir):
        dataset = NIHDenoisingDataset(test_dir, image_size=256, noise_factor=0.05)
        noisy, clean = dataset[0]
        print(f"Dataset size: {len(dataset)}")
        print(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
        print("Dataset script is working properly.")
    else:
        print(f"Directory {test_dir} not found. Check your path.")

    # show the noisy and clean images for the first sample
    import matplotlib.pyplot as plt

    plt.switch_backend("Qt5Agg")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy.squeeze().numpy(), cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(clean.squeeze().numpy(), cmap='gray')
    plt.title("Clean Image")
    plt.axis('off')
    plt.show()
