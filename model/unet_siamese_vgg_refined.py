import torch
import torch.nn as nn
import torchvision.models as models



class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dropout_rate = 0.1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)

class VGG11SiameseUnet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        super(VGG11SiameseUnet, self).__init__()

        self.conv1 = DownsampleBlock(n_channels,64)
        self.conv2 = DownsampleBlock(64,128)
        self.conv3s = DownsampleBlock(128,256)
        self.conv3 = DownsampleBlock(256,256)
        self.conv4s = DownsampleBlock(256,512)
        self.conv4 = DownsampleBlock(512,512)
        self.conv5s = DownsampleBlock(512,512)
        self.conv5 = DownsampleBlock(512,512)
        self.pool = nn.MaxPool2d(2)
        self.center = UpSampleBlock(512, 512, 256)
        self.dec5 = UpSampleBlock(512+256, 512, 256)
        self.dec4 = UpSampleBlock(512+256, 512, 128)
        self.dec3 = UpSampleBlock(256+128, 256, 64)
        self.dec2 = UpSampleBlock(128+64, 128, 32)
        self.dec1 = nn.Sequential(nn.Conv2d(64+32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))#ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, F_t1, F_t, M_t1):
        # F_t1 ... F(t-1)
        # F_t ... F(t)
        # M_t1 ... M(t-1)
        F_t1_with_mask = torch.cat((F_t1, M_t1), dim=1)
        F_t_with_mask = torch.cat((F_t, M_t1), dim=1)
        
        ## Encoder F(t-1)
        conv1_1 = self.conv1(F_t1_with_mask)
        conv2_1 = self.conv2(self.pool(conv1_1))
        conv3s_1 = self.conv3s(self.pool(conv2_1))
        conv3_1 = self.conv3(conv3s_1)
        conv4s_1 = self.conv4s(self.pool(conv3_1))
        conv4_1 = self.conv4(conv4s_1)
        conv5s_1 = self.conv5s(self.pool(conv4_1))
        conv5_1 = self.conv5(conv5s_1)
        
        ## Encoder F(t)
        conv1_2 = self.conv1(F_t_with_mask)
        conv2_2 = self.conv2(self.pool(conv1_2))
        conv3s_2 = self.conv3s(self.pool(conv2_2))
        conv3_2 = self.conv3(conv3s_2)
        conv4s_2 = self.conv4s(self.pool(conv3_2))
        conv4_2 = self.conv4(conv4s_2)
        conv5s_2 = self.conv5s(self.pool(conv4_2))
        conv5_2 = self.conv5(conv5s_2)

        ## Fuze encoders
        # Averaging
        x5 = 0.5 * conv5_1 + 0.5 * conv5_2
        
        ## Bottleneck
        center = self.center(self.pool(x5))

        ## Decoder (we want to segment F(t))        
        x = self.dec5(torch.cat([center, conv5_2], 1))
        x= self.dec4(torch.cat([x, conv4_2], 1))
        x = self.dec3(torch.cat([x, conv3_2], 1))
        x = self.dec2(torch.cat([x, conv2_2], 1))
        x = self.dec1(torch.cat([x, conv1_2], 1))

        logits = self.final(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities
