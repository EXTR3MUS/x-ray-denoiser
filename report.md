# Technical Report: Chest X-Ray Denoising

## 1. Problem Definition
The quality of medical images is a critical factor for diagnostic accuracy. In X-ray examinations, factors such as radiation dose reduction (for patient safety) or mechanical limitations of the sensors frequently introduce noise during capture. This project aims to solve this problem by building a Deep Learning system capable of denoising chest radiographs, restoring the clarity of anatomical structures without degrading essential clinical details.

## 2. Justification of the Chosen Approach
The denoising task was selected over anomaly classification approaches for technical reasons and structural feasibility. Real-world medical datasets suffer from severe class imbalance (the vast majority of exams are healthy) and frequently imprecise labels, which would require extensive data engineering to avoid biased models.

By adopting denoising, we transform the problem into a pixel-to-pixel mapping task with a perfect and absolute Ground Truth. We use clean radiographs as targets and synthetically generate noisy images as inputs.

For the required experimentation flow, the project architecture opposes two families of models for critical analysis:
* **CNN Model (Baseline):** A **U-Net** architecture developed from scratch. Chosen for being the gold standard in medical image-to-image translation tasks, utilizing skip connections to preserve the high frequency and spatial resolution of anatomical edges.
* **Attention-Based Model (Variation):** An architecture based on **Transformers** focused on image restoration (e.g., fine-tuned Swin-Unet or SwinIR). The choice aims to evaluate whether the global attention capacity of Transformers surpasses the local inductive bias (local receptive fields) of traditional convolutional networks in this context.

## 3. Data Strategy and Preprocessing
The base data originates from the **NIH Chest X-ray Dataset**. The processing pipeline was designed to maximize I/O and local GPU processing efficiency (12GB VRAM limit):
* **Dynamic Pipeline:** Loading (via a custom PyTorch `Dataset` class) converts the images to grayscale and applies downsampling to a 256x256 pixel resolution, enabling the fast processing of larger batches.
* **On-the-fly Augmentation and Noise:** Instead of storing static pairs of clean/noisy images, Gaussian noise is mathematically generated and added to the tensors at runtime (on-the-fly). This ensures that, at every training epoch, the model is exposed to a different noise pattern for the same image, acting natively as data augmentation and drastically reducing the chances of overfitting.