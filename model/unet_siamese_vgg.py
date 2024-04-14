import torch
import torch.nn as nn
import torchvision.models as models

# Inspiration: https://arxiv.org/pdf/1801.05746v1.pdf, https://github.com/ternaus/TernausNet/blob/master/ternausnet/models.py
# + LucidTracker

def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.Dropout(0.5),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VGG11SiameseUnet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, num_filters=32):
        super(VGG11SiameseUnet, self).__init__()

        # Load preatrained VGG11, dont load the classifier part
        vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        # Modify the first layer so it accepts 4 channels
        first_conv = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.vgg11 = nn.Sequential(
            first_conv,
            *vgg11[1:]  # Rest of the VGG11 without the first layer
        )

        dropout_rate = 0.0

        self.conv1 = nn.Sequential(self.vgg11[0], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv2 = nn.Sequential(self.vgg11[3], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv3s = nn.Sequential(self.vgg11[6], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv3 = nn.Sequential(self.vgg11[8], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv4s = nn.Sequential(self.vgg11[11], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv4 = nn.Sequential(self.vgg11[13], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv5s = nn.Sequential(self.vgg11[16], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.conv5 = nn.Sequential(self.vgg11[18], nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.pool = nn.MaxPool2d(2)

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, F_t1, F_t, M_t1):
        F_t1_with_mask = torch.cat((F_t1, M_t1), dim=1)
        F_t_with_mask = torch.cat((F_t, M_t1), dim=1)
        
        # Encoder F(t-1)
        conv1_1 = self.conv1(F_t1_with_mask)
        conv2_1 = self.conv2(self.pool(conv1_1))
        conv3s_1 = self.conv3s(self.pool(conv2_1))
        conv3_1 = self.conv3(conv3s_1)
        conv4s_1 = self.conv4s(self.pool(conv3_1))
        conv4_1 = self.conv4(conv4s_1)
        conv5s_1 = self.conv5s(self.pool(conv4_1))
        conv5_1 = self.conv5(conv5s_1)
        
        # Encoder F(t)
        conv1_2 = self.conv1(F_t_with_mask)
        conv2_2 = self.conv2(self.pool(conv1_2))
        conv3s_2 = self.conv3s(self.pool(conv2_2))
        conv3_2 = self.conv3(conv3s_2)
        conv4s_2 = self.conv4s(self.pool(conv3_2))
        conv4_2 = self.conv4(conv4s_2)
        conv5s_2 = self.conv5s(self.pool(conv4_2))
        conv5_2 = self.conv5(conv5s_2)

        # Fuze encoders
        #x5 = torch.cat((x5_1, x5_2), dim=1)
        x5 = 0.5 * conv5_1 + 0.5 * conv5_2
        #print(x5.size())
        # Bottleneck
        center = self.center(self.pool(x5))

        # Decoder (we want to segment F(t))        
        x = self.dec5(torch.cat([center, conv5_2], 1))
        x= self.dec4(torch.cat([x, conv4_2], 1))
        x = self.dec3(torch.cat([x, conv3_2], 1))
        x = self.dec2(torch.cat([x, conv2_2], 1))
        x = self.dec1(torch.cat([x, conv1_2], 1))

        logits = self.final(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities
