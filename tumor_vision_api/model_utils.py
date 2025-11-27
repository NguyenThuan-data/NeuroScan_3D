import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(MultiTaskUNet, self).__init__()
        
        # ENCODER (Left Side)
        self.enc1 = self.conv_block(in_channels, 16)## change the 16 to 32 -> 64 -> 128, the cost is memory intensive
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.bottleneck = self.conv_block(64, 128)
        
        # Max Pooling (Downsampling)
        self.pool = nn.MaxPool3d(2)
        
        # CLASSIFICATION 
        # Branches off from the bottleneck
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(0.5), ##If Training Loss is low but Validation Loss is high: You are overfitting. Increase Dropout to 0.6 or 0.7.
##If Training Loss stays high: The model is too weak. Decrease Dropout to 0.2 or 0.3.
            nn.Linear(64, 1)
        )
        
        #  SEGMENTATION (Decoder / Right Side)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64) # 64+64=128 input channels due to cat
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32) # 32+32=64 input
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16) # 16+16=32 input
        
        self.final_seg = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Classification
        # Flatten the bottleneck and guess HGG/LGG
        cls = self.classifier(self.global_pool(b).view(b.size(0), -1))
        
        # Segmentation (U-Net Path)
        # Level 3 Up
        up3 = self.up3(b)
        # SKIP CONNECTION: Concatenate (cat) Up-sampled feature + E3 feature
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        
        # Level 2 Up
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        
        # Level 1 Up
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))
        
        seg = self.final_seg(d1)
        
        return seg, cls

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            # Double Conv is standard in U-Net
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

# Initialize
# model = MultiTaskUNet().to(device)
# print("Multi-Task U-Net Created")


class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1.):
        pred = torch.softmax(pred, dim=1)
        dice = 0
        # Calculate Dice for classes 1, 2, 3 (Ignore background 0)
        for cls in [1, 2, 3]:
            p_flat = pred[:, cls].contiguous().view(-1)
            t_flat = (target == cls).float().contiguous().view(-1)
            intersection = (p_flat * t_flat).sum()
            dice += (2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)
        return 1 - (dice / 3)

## is to tell which loss is more important
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, seg_p, seg_t, cls_p, cls_t):
        return self.dice(seg_p, seg_t) + 0.2 * self.bce(cls_p, cls_t)