import torch
import torch.nn as nn

# -------------------------------Unet-MODEL-------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.up1 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)

        x = self.upsample(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)
        logits = self.outc(x)
        probabilities = torch.sigmoid(logits)  # Převedení logitů na pravděpodobnosti
        return probabilities
