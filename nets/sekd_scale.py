
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(torch.nn.Module):
    """ChannelAttention: a module to estimate the weight of each channel.
    """
    def __init__(self, channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features=channels,
                      out_features=int(channels/ratio), bias=True),
            nn.ReLU(),
            nn.Linear(in_features=int(channels/ratio),
                      out_features=channels, bias=True),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, input):
        pooled_features = self.pool(input).squeeze(3).squeeze(2)
        attention_weight = self.attention(pooled_features)
        attention_weight = attention_weight.unsqueeze(2).unsqueeze(2)
        weighted_features = input * attention_weight
        return weighted_features

class ResBlock(torch.nn.Module):
    """ConvResBlock: define a convolutional block with residual shortcut.
       It is the caller's duty to ensure the same size of input and output.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros'):
        super(ResBlock, self).__init__()
        self.convb = nn.Sequential(
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True, padding_mode=padding_mode),
        )
        self.conv_residual = None
        if stride != 1 or in_channels != out_channels:
            self.conv_residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, bias=False)
        return

    def forward(self, input):
        if self.conv_residual != None:
            residual = self.conv_residual(input)
        else:
            residual = input
        out = self.convb(input)
        out += residual
        return out

class SEKDScale(torch.nn.Module):
    """SEDKScale model definition, simultaneously detect and describe keypoints.
    """
    def __init__(self):
        super(SEKDScale, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True)
        self.resblock0 = nn.Sequential(
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
        )

        self.resblock1 = nn.Sequential(
            ResBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
            ResBlock(64, 32, kernel_size=3, padding=1),
        )

        self.resblock2 = nn.Sequential(
            ResBlock(32, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 32, kernel_size=3, padding=1),
        )

        self.resblock3 = nn.Sequential(
            ResBlock(32, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 32, kernel_size=3, padding=1),
        )

        self.detector_attention = ChannelAttention(channels=128)
        self.detector = nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=True)

        self.descriptor_attention = ChannelAttention(channels=128)
        self.descriptor = ResBlock(128, 128, kernel_size=1)

        return

    def forward(self, input):
        feature0 = self.resblock0(self.conv0(input))
        feature1_ = self.resblock1(feature0)
        feature2_ = self.resblock2(feature1_)
        feature3_ = self.resblock3(feature2_)

        feature1 = F.interpolate(feature1_, scale_factor=2, mode='bilinear')
        feature2 = F.interpolate(feature2_, scale_factor=4, mode='bilinear')
        feature3 = F.interpolate(feature3_, scale_factor=8, mode='bilinear')

        features = torch.cat((feature0, feature1, feature2, feature3), dim=1)

        score = self.detector(self.detector_attention(features))
        descriptor = self.descriptor(self.detector_attention(features))

        return score, descriptor

