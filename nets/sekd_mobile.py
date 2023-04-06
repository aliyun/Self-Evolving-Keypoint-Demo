
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlockMobile(torch.nn.Module):
    """ConvResBlockMobile: define a convolutional block with residual shortcut.
       It is the caller's duty to ensure the same size of input and output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups_d=1, groups_c=1, groups_r=1,
                 padding_mode='zeros'):
        super(ResBlockMobile, self).__init__()
        self.convb = nn.Sequential(
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups_d, bias=False, padding_mode=padding_mode),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1,
                stride=1, padding=0, dilation=1,
                groups=groups_c, bias=True),
        )
        self.conv_residual = None
        if stride != 1 or in_channels != out_channels:
            self.conv_residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, groups=groups_r, bias=False)
        return

    def forward(self, input):
        if self.conv_residual != None:
            residual = self.conv_residual(input)
        else:
            residual = input
        out = self.convb(input)
        out += residual
        return out

class SEKDMobile(torch.nn.Module):
    """SEDKMobile model definition, simultaneously detect and describe keypoints.
    """
    def __init__(self):
        super(SEKDMobile, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        self.resblock0 = nn.Sequential(
            ResBlockMobile(32, 32, kernel_size=3, padding=1,
                           groups_d=32, groups_c=8),
            ResBlockMobile(32, 32, kernel_size=3, padding=1,
                           groups_d=32, groups_c=8),
            ResBlockMobile(32, 32, kernel_size=3, padding=1,
                           groups_d=32, groups_c=1),
        )

        self.resblock1 = nn.Sequential(
            ResBlockMobile(32, 64, kernel_size=3, stride=2, padding=1,
                           groups_d=32, groups_c=16, groups_r=8),
            ResBlockMobile(64, 64, kernel_size=3, padding=1,
                           groups_d=64, groups_c=16),
            ResBlockMobile(64, 64, kernel_size=3, padding=1,
                           groups_d=64, groups_c=1),
        )

        self.resblock2 = nn.Sequential(
            ResBlockMobile(64, 128, kernel_size=3, stride=2, padding=1,
                           groups_d=64, groups_c=32, groups_r=16),
            ResBlockMobile(128, 128, kernel_size=3, padding=1,
                           groups_d=128, groups_c=32),
            ResBlockMobile(128, 128, kernel_size=3, padding=1,
                           groups_d=128, groups_c=1),
        )

        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                               groups=64, output_padding=1, bias=True),
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               groups=32, output_padding=1, bias=True),
        )

        self.detector = nn.Conv2d(
            32, 2, kernel_size=3, padding=1, groups=2, bias=True)

        self.descriptor = ResBlockMobile(
            128, 128, kernel_size=3, padding=1, groups_d=128, groups_c=32)

        return

    def forward(self, input):
        feature0_ = self.resblock0(self.conv0(input))
        feature1_ = self.resblock1(feature0_)
        feature2 = self.resblock2(feature1_)

        feature1 = feature1_ + self.deconv0(feature2)
        feature0 = feature0_ + self.deconv1(feature1)

        score = self.detector(feature0)
        descriptor = self.descriptor(feature2)

        return score, descriptor, feature0

