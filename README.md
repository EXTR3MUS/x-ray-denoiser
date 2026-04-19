# x-ray-denoiser
X-Ray denoising experiment comparing U-Net and Transformer based architectures.

## Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## File structure

```
x-ray-denoiser/
├── data/                   # Raw and processed images (git ignored)
├── images/                 # Saved plots and visual results for the report
├── src/
│   ├── dataset.py          # Dataloader and synthetic noise injection
│   ├── models/
│   │   ├── unet.py         # Baseline U-Net
│   │   └── transformer.py  # SwinIR/Transformer implementation
│   ├── train.py            # Training loop with AMP
│   └── utils.py            # PSNR/SSIM metrics and matplotlib savers
├── report.md               # The 1,500-3,000 word text
├── requirements.txt
└── .gitignore
```