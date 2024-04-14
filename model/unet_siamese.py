import torch
import torch.nn as nn

class ConvDoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # convolution 3x3 => [BN] => ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.down(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.convD = ConvDoubleBlock(in_channels, out_channels)

    def forward(self, x, xn):
        x = self.up(x)
        x = torch.cat([x, xn], dim=1)
        x = self.dropout(x)
        x = self.convD(x)
        return x

class Siamese(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        super(Siamese, self).__init__()
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

    def forward(self, F_t1, F_t, M_t1):
        F_t1_with_mask = torch.cat((F_t1, M_t1), dim=1)
        F_t_with_mask = torch.cat((F_t, M_t1), dim=1)
        
        # Encoder F(t-1)
        x1_1 = self.inc(F_t1_with_mask)
        x2_1 = self.down1(x1_1) # size/2
        x3_1 = self.down2(x2_1) # size/4
        x4_1 = self.down3(x3_1) # size/8
        x5_1 = self.down4(x4_1) # size/16
        
        # Encoder F(t)
        x1_2 = self.inc(F_t_with_mask)
        x2_2 = self.down1(x1_2) # size/2
        x3_2 = self.down2(x2_2) # size/4
        x4_2 = self.down3(x3_2) # size/8
        x5_2 = self.down4(x4_2) # size/16

        # Fuze encoders
        #x5 = torch.cat((x5_1, x5_2), dim=1)
        x5 = 0.5 * x5_1 + 0.5 * x5_2
        #print(x5.size())
        # Bottleneck
        bottleneck = self.bottleneck(x5)

        # Decoder (we want to segment F(t))
        x = self.up4(bottleneck,x4_2) # size * 2
        x = self.up3(x,x3_2) # size * 4
        x = self.up2(x,x2_2) # size * 8
        x = self.up1(x,x1_2) # size * 16
        
        logits = self.outc(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities
