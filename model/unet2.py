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

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            ConvDoubleBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.down(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convD = ConvDoubleBlock(in_channels, out_channels)

    def forward(self, x, xn):
        x = self.up(x)
        x = torch.cat([x, xn], dim=1)
        x = nn.Dropout(0.3)(x)
        x = self.convD(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = ConvDoubleBlock(n_channels, 64)
        
        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)
        self.down3 = DownsampleBlock(256, 512)
        self.down4 = DownsampleBlock(512, 1024)
        
        self.bottleneck = ConvDoubleBlock(1024, 1024)
        
        self.up4 = UpsampleBlock(1024, 512)
        self.up3 = UpsampleBlock(512, 256)
        self.up2 = UpsampleBlock(256, 128)
        self.up1 = UpsampleBlock(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, inputs):
        # Encoder
        x1 = self.inc(inputs)
        x2 = self.down1(x1) # size/2
        x3 = self.down2(x2) # size/4
        x4 = self.down3(x3) # size/8
        x5 = self.down4(x4) # size/16

        # Bottleneck
        bottleneck = self.bottleneck(x5)

        # Decoder
        x = self.up4(bottleneck,x4) # size * 2
        x = self.up3(x,x3) # size * 4
        x = self.up2(x,x2) # size * 8
        x = self.up1(x,x1) # size * 16
        
        logits = self.outc(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities
