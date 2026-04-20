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

## 4. Experiment Structure and Baseline Architecture (U-Net)

To adapt the project scope to the available time and computational resources (local training using a 12GB RTX 3060 GPU), the dataset subset was defined at 1,500 images. The dataset was strictly partitioned into:
* **Training:** 1,000 images (66%)
* **Validation:** 250 images (17%)
* **Testing:** 250 images (17%)

This proportion guarantees sufficient volume for the convolutional filters to learn while reserving isolated data for cross-validation during training and final metric evaluation.

The Baseline model implementation consisted of a **U-Net built from scratch** using PyTorch. The architecture was configured with 4 levels of downsampling (Max Pooling) followed by double convolutions, a bottleneck block, and 4 levels of upsampling (ConvTranspose2d) connected via skip connections with the corresponding encoder layers. The output layer utilizes a Sigmoid activation function, ensuring that the predicted pixels map perfectly to the `[0.0, 1.0]` range, corresponding to the same normalized scale (enforced via `torch.clamp`) as the noisy input images.

## 5. Metrics and Training Strategy

The model was optimized using the **Adam** algorithm with a learning rate of $1 \times 10^{-3}$. The Mean Squared Error (**MSE**) was chosen as the Loss Function, which heavily penalizes large pixel deviations between the generated image and the Ground Truth image. 

To accelerate training and mitigate video memory constraints, the training loop was wrapped using PyTorch's native Automatic Mixed Precision (**AMP**) via `torch.amp.autocast`. This performs critical tensor calculations in 16-bit float precision (FP16) where supported, significantly speeding up backpropagation on the Ampere architecture.

For the qualitative and quantitative analysis of the results, alongside the MSE loss curve, the project adopted the **PSNR (Peak Signal-to-Noise Ratio)** metric. Unlike MSE, PSNR is expressed in decibels (dB) and provides a standardized measure of image reconstruction quality. Higher PSNR values indicate that the model successfully removed the noise with minimal degradation of the original anatomical signal.

## 6. Architectural Variation: Attention U-Net

To fulfill the experimentation requirement and critically analyze different approaches, a second architecture was implemented: the **Attention U-Net**. 

While pure Vision Transformers (ViTs) provide powerful global context, they lack the inductive bias of convolutions, often requiring massive datasets and prolonged training times to converge—factors that are prohibitive in rapid prototyping scenarios. The Attention U-Net serves as a highly efficient hybrid solution. It maintains the robust convolutional backbone of the standard U-Net while integrating Attention Gates (AGs) into the skip connections.

**Justification for the Hybrid Approach:**
In standard U-Nets, skip connections concatenate high-resolution feature maps from the encoder directly into the decoder. This introduces redundant, low-level background features (such as empty space around the patient's torso) that the model must expend computational effort to suppress. 

The Attention Block mathematically addresses this by utilizing the gating signal from the decoder to "prune" the skip connection features before concatenation. The attention mechanism generates a spatial gating multiplier (ranging from 0 to 1 via a Sigmoid activation) that highlights salient anatomical structures (like the rib cage and lungs) while suppressing irrelevant background noise. This allows the model to leverage the conceptual benefits of attention (focusing computational resources on the most relevant spatial regions) without abandoning the parameter efficiency and rapid convergence of convolutional layers.

## 7. Critical Analysis of Results

The evaluation of the models revealed significant insights into the training dynamics of deep learning architectures under constrained data regimes. 

Analyzing the baseline U-Net learning curves (Loss and PSNR), the model demonstrated a rapid initial convergence, with the training MSE dropping significantly within the first two epochs. However, the validation metrics exhibited high volatility. The validation PSNR fluctuated sharply between 28.0 dB and 31.4 dB, with corresponding spikes in the validation loss. 

This erratic behavior can be attributed to two main factors:
1. **Optimizer Overshooting:** The static learning rate of $10^{-3}$ proved optimal for initial weight adjustments but was too aggressive for fine-tuning the convolutional filters in later epochs, causing the Adam optimizer to oscillate around the local minima rather than converging smoothly. Future iterations would benefit from a learning rate scheduler (e.g., `ReduceLROnPlateau`).
2. **Sample Variance:** The restricted validation set (250 images) resulted in high variance per evaluation step. A small number of batches with complex anatomical structures or high-density noise disproportionality impacted the epoch averages.

Despite the volatility, the model successfully learned the underlying noise distribution, proving the viability of the dynamic *on-the-fly* noise injection strategy. The Attention U-Net variation theoretically mitigates some of this instability by using its gating mechanisms to suppress irrelevant background noise, allowing the network to focus its parameter updates strictly on the pulmonary and skeletal structures, potentially leading to smoother generalization.

## 8. Production Deployment Architecture

Transitioning this model from a local Jupyter/Python environment into a robust, real-world clinical application requires decoupling the heavy machine learning inference from the user-facing systems. 

A modern, scalable production architecture would be structured as a microservices ecosystem:

1. **Core Backend (TypeScript/Node.js):** The main application logic should be handled by a robust TypeScript backend. This service manages user authentication (clinical staff), authorization, metadata storage in a relational database (e.g., PostgreSQL), and orchestrates the incoming radiograph files.
2. **Asynchronous Inference Pipeline:** Medical image inference is computationally expensive and must not block the main backend threads. When the TypeScript backend receives an X-ray, it should upload the raw file to an object storage bucket (e.g., AWS S3) and publish a message to a queue/broker (such as RabbitMQ or Redis).
3. **Model Microservice (Python/FastAPI):** A dedicated, containerized Python microservice (running on GPU-enabled nodes) listens to the queue. It fetches the image, runs the PyTorch U-Net inference, saves the denoised output back to the object storage, and sends a webhook or event back to the main TypeScript backend signaling completion.
4. **Containerization & Scaling:** Both the backend and the inference microservice must be containerized using Docker. The inference service can be deployed on a Kubernetes cluster with GPU autoscaling, spinning up additional Pods dynamically when the queue of X-rays to be processed grows during peak clinical hours.