import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_shortcut = self.bn_shortcut(self.conv_shortcut(x))

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + x_shortcut
        x = F.relu(x)
        return x

def upsample_concat(x, skip):
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = torch.cat((x, skip), dim=1)
    return x

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ResBlock(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ResBlock(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ResBlock(64, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ResBlock(128, 256)

        # Decoder
        self.up1 = ResBlock(256 + 128, 128)
        self.up2 = ResBlock(128 + 64, 64)
        self.up3 = ResBlock(64 + 32, 32)
        self.up4 = ResBlock(32 + 16, 16)

        # Final Output Layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        pool1_out = self.pool1(conv1_out)

        conv2_out = self.conv2(pool1_out)
        pool2_out = self.pool2(conv2_out)

        conv3_out = self.conv3(pool2_out)
        pool3_out = self.pool3(conv3_out)

        conv4_out = self.conv4(pool3_out)
        pool4_out = self.pool4(conv4_out)

        # Bottleneck
        conv5_out = self.conv5(pool4_out)

        # Decoder
        up1_out = upsample_concat(conv5_out, conv4_out)
        up1_out = self.up1(up1_out)

        up2_out = upsample_concat(up1_out, conv3_out)
        up2_out = self.up2(up2_out)

        up3_out = upsample_concat(up2_out, conv2_out)
        up3_out = self.up3(up3_out)

        up4_out = upsample_concat(up3_out, conv1_out)
        up4_out = self.up4(up4_out)

        # Final output
        output = torch.sigmoid(self.final_conv(up4_out))
        return output