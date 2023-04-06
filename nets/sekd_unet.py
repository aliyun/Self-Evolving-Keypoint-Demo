
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(torch.nn.Module):
    """ConvResBlock: define a convolutional block with residual shortcut.
       It is the caller's duty to ensure the same size of input and output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(ResBlock, self).__init__()
        self.convb = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True, padding_mode=padding_mode),
        )
        self.conv_residual = None
        if stride != 1 or in_channels != out_channels:
            self.conv_residual = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0, bias=False)
        return

    def forward(self, input):
        if self.conv_residual != None:
            residual = self.conv_residual(input)
        else:
            residual = input
        out = self.convb(input)
        out += residual
        return out

class SEKDUNet(torch.nn.Module):
    """SEDKUNet model definition, simultaneously detect and describe keypoints.
    """
    def __init__(self):
        super(SEKDUNet, self).__init__()
        self.ratio = 16
        self.resblock0_ = nn.Sequential(
            ResBlock(1, 64, kernel_size=5, padding=2),
        )

        self.resblock1_ = nn.Sequential(
            ResBlock(64, 64, kernel_size=5, stride=2, padding=2),
        )

        self.resblock2_ = nn.Sequential(
            ResBlock(64, 128, kernel_size=5, stride=2, padding=2),
        )

        self.resblock3_ = nn.Sequential(
            ResBlock(128, 128, kernel_size=5, stride=2, padding=2),
        )

        self.resblock4_ = nn.Sequential(
            ResBlock(128, 128, kernel_size=5, stride=2, padding=2),
        )

        self.deconv3 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=5, stride=2, padding=2,
                output_padding=1, bias=True),
        )

        self.resblock3 = nn.Sequential(
            ResBlock(256, 128, kernel_size=5, padding=2),
        )

        self.deconv2 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=5, stride=2, padding=2,
                output_padding=1, bias=True),
        )

        self.resblock2 = nn.Sequential(
            ResBlock(256, 128, kernel_size=5, padding=2),
        )

        self.deconv1 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2,
                output_padding=1, bias=True),
        )

        self.resblock1 = nn.Sequential(
            ResBlock(128, 64, kernel_size=5, padding=2),
        )

        self.deconv0 = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 64, kernel_size=5, stride=2, padding=2,
                output_padding=1, bias=True),
        )

        self.resblock0 = nn.Sequential(
            ResBlock(128, 64, kernel_size=5, padding=2),
        )

        self.detector = nn.Conv2d(64, 2, kernel_size=3, padding=1, bias=True)

        self.descriptor = ResBlock(128, 128, kernel_size=5, padding=2)

        return

    def forward(self, input):
        _, _, height, width = input.shape
        assert(((height % self.ratio) == 0) and ((width % self.ratio) == 0))

        feature0_ = self.resblock0_(input)
        feature1_ = self.resblock1_(feature0_)
        feature2_ = self.resblock2_(feature1_)
        feature3_ = self.resblock3_(feature2_)
        feature4 = self.resblock4_(feature3_)

        feature3 = self.resblock3(
            torch.cat((feature3_, self.deconv3(feature4)), dim = 1))
        feature2 = self.resblock2(
            torch.cat((feature2_, self.deconv2(feature3)), dim = 1))
        feature1 = self.resblock1(
            torch.cat((feature1_, self.deconv1(feature2)), dim = 1))
        feature0 = self.resblock0(
            torch.cat((feature0_, self.deconv0(feature1)), dim = 1))

        score = self.detector(feature0)

        descriptor = self.descriptor(feature2)

        return score, descriptor, feature0

