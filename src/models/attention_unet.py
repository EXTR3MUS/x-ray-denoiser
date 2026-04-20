"""Attention U-Net model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .unet import DoubleConv
except ImportError:
    # Support direct execution: python src/models/attention_unet.py
    from unet import DoubleConv

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(AttentionUNet, self).__init__()
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with Attention
        g1 = self.up1(x5)
        x4_att = self.att1(g=g1, x=x4)
        d1 = torch.cat((x4_att, g1), dim=1)
        d1 = self.conv1(d1)
        
        g2 = self.up2(d1)
        x3_att = self.att2(g=g2, x=x3)
        d2 = torch.cat((x3_att, g2), dim=1)
        d2 = self.conv2(d2)
        
        g3 = self.up3(d2)
        x2_att = self.att3(g=g3, x=x2)
        d3 = torch.cat((x2_att, g3), dim=1)
        d3 = self.conv3(d3)
        
        g4 = self.up4(d3)
        x1_att = self.att4(g=g4, x=x1)
        d4 = torch.cat((x1_att, g4), dim=1)
        d4 = self.conv4(d4)
        
        logits = self.outc(d4)
        return self.sigmoid(logits)
    
if __name__ == '__main__':
    # Test tensor: Batch Size 2, 1 Channel, 256x256
    x = torch.randn((2, 1, 256, 256))
    model = AttentionUNet(n_channels=1, n_classes=1)
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")