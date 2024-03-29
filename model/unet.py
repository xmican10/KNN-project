import torch
import torch.nn as nn

# -------------------------------Unet-MODEL-------------------------------------
"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        self.inc = ConvDoubleBlock(n_channels, 64)
        
        self.down1 = ConvDoubleBlock(64, 128)
        self.down2 = ConvDoubleBlock(128, 256)
        self.down3 = ConvDoubleBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = ConvDoubleBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvDoubleBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ConvDoubleBlock(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        
        x2 = self.pool(x1) # size/2
        x2 = self.down1(x2)
        
        x3 = self.pool(x2) # size/4
        x3 = self.down2(x3)
        
        x4 = self.pool(x3) # size/8
        x4 = self.down3(x4)

        # Decoder
        x = self.up1(x4) # size * 2
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x) # size * 4
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x) # size * 8
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
"""
# Optimal for 128x128 input

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = ConvDoubleBlock(n_channels, 64)
        self.down1 = ConvDoubleBlock(64, 128)
        self.down2 = ConvDoubleBlock(128, 256)
        self.up1 = ConvDoubleBlock(256 + 128, 128)
        self.up2 = ConvDoubleBlock(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        # size
        x1 = self.inc(x)
        x2 = self.pool(x1)
        # size/2
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        # size/4
        x3 = self.down2(x3)

        # Decoder
        x = self.upsample(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)
        logits = self.outc(x)
        probabilities = torch.sigmoid(logits)
        return probabilities
