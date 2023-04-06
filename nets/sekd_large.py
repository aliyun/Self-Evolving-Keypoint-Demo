
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

class SEKDLarge(torch.nn.Module):
    """SEDK large model definition.
    """
    def __init__(self):
        super(SEKDLarge, self).__init__()
        self.ratio = 4
        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True)
        self.resblock0 = nn.Sequential(
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
        )

        self.resblock1 = nn.Sequential(
            ResBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
        )

        self.resblock2 = nn.Sequential(
            ResBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
        )

        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1,
                output_padding=1, bias=True),
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1,
                output_padding=1, bias=True),
        )

        self.detector = nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=True)

        self.descriptor = ResBlock(128, 128, kernel_size=3, padding=1)

        return

    def forward(self, input_init):
        _, _, height, width = input_init.shape
        if ((height % self.ratio) == 0) and ((width % self.ratio) == 0):
            height_input = height
            width_input = width
            input = input_init
        else:
            height_input = int(height / self.ratio) * self.ratio
            width_input = int(width / self.ratio) * self.ratio
            input = input_init[:, :, 0:height_input, 0:width_input]

        feature0_ = self.resblock0(self.conv0(input))
        feature1_ = self.resblock1(feature0_)
        feature2 = self.resblock2(feature1_)

        feature1 = feature1_ + self.deconv0(feature2)
        feature0 = feature0_ + self.deconv1(feature1)

        score = self.detector(feature0)

        descriptor = self.descriptor(feature2)
        descriptor = F.interpolate(
            descriptor, (height_input, width_input), mode='bilinear')

        if not (((height % self.ratio) == 0) and ((width % self.ratio) == 0)):
            score = F.pad(
                score, (0, width - width_input, 0, height - height_input))
            descriptor = F.pad(
                descriptor,
                (0, width - width_input, 0, height - height_input),
                value = 1. / (descriptor.shape[1] ** 0.5))
            feature0 = F.pad(
                feature0,
                (0, width - width_input, 0, height - height_input),
                value = 1. / (feature0.shape[1] ** 0.5))

        return score, descriptor, feature0

