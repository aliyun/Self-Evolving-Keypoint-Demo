
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

class SEKDMobileCV2(torch.nn.Module):
    """Definition of SEDKMobile model for CV2, simultaneously detect and
       describe keypoints.
    """
    def __init__(self):
        super(SEKDMobileCV2, self).__init__()
        self.conv0 = nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=True)
        self.resblock0 = nn.Sequential(
            ResBlockMobile(24, 24, kernel_size=3, padding=1,
                           groups_d=24, groups_c=1),
            ResBlockMobile(24, 24, kernel_size=3, padding=1,
                           groups_d=24, groups_c=1),
            ResBlockMobile(24, 24, kernel_size=3, padding=1,
                           groups_d=24, groups_c=1),
        )

        self.resblock1 = nn.Sequential(
            ResBlockMobile(24, 48, kernel_size=3, stride=2, padding=1,
                           groups_d=24, groups_c=1, groups_r=1),
            ResBlockMobile(48, 48, kernel_size=3, padding=1,
                           groups_d=48, groups_c=1),
            ResBlockMobile(48, 48, kernel_size=3, padding=1,
                           groups_d=48, groups_c=1),
        )

        self.resblock2 = nn.Sequential(
            ResBlockMobile(48, 96, kernel_size=3, stride=2, padding=1,
                           groups_d=48, groups_c=1, groups_r=1),
            ResBlockMobile(96, 96, kernel_size=3, padding=1,
                           groups_d=96, groups_c=1),
            ResBlockMobile(96, 96, kernel_size=3, padding=1,
                           groups_d=96, groups_c=1),
        )

        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(96, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1,
                      groups=96, bias=True),
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True),
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(48, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1,
                      groups=48, bias=True),
            nn.Conv2d(48, 24, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True),
        )

        self.detector = nn.Conv2d(
            24, 2, kernel_size=3, padding=1, groups=1, bias=True)

        self.descriptor = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1,
                      groups=96, bias=True),
            nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=True),
        )

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

class InvertedResBlockMobile(torch.nn.Module):
    """InvertedResBlockMobile: define a convolutional block with
       residual shortcut.
       It is the caller's duty to ensure the same size of input and output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, padding_mode='zeros', expansion=1):
        super(InvertedResBlockMobile, self).__init__()
        if expansion > 1:
            self.convb = nn.Sequential(
                nn.BatchNorm2d(in_channels, affine=True),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels, in_channels*expansion, kernel_size=1,
                    stride=1, padding=0, dilation=1, groups=1, bias=True),
                nn.Conv2d(
                    in_channels*expansion, in_channels*expansion,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation,
                    groups=in_channels*expansion, bias=False,
                    padding_mode=padding_mode),
                nn.Conv2d(
                    in_channels*expansion, out_channels, kernel_size=1,
                    stride=1, padding=0, dilation=1, groups=1, bias=True),
            )
        else:
            self.convb = nn.Sequential(
                nn.BatchNorm2d(in_channels, affine=True),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False, padding_mode=padding_mode),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, dilation=1, groups=1, bias=True),
            )
        self.shortcut = False
        if stride == 1 and in_channels == out_channels:
            self.shortcut = True
        return

    def forward(self, input):
        out = self.convb(input)
        if self.shortcut == True:
            out += input
        return out

class SEKDMobile2CV2(torch.nn.Module):
    """Definition of SEDKMobile2 model for CV2, simultaneously detect and
       describe keypoints.
       Inference time: 9 ms.
    """
    def __init__(self):
        super(SEKDMobile2CV2, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.resblock0 = nn.Sequential(
            InvertedResBlockMobile(
                16, 16, kernel_size=3, padding=1, expansion=2),
            InvertedResBlockMobile(
                16, 16, kernel_size=5, padding=2),
            InvertedResBlockMobile(
                16, 16, kernel_size=3, padding=1, expansion=2),
        )

        self.resblock1 = nn.Sequential(
            InvertedResBlockMobile(
                16, 24, kernel_size=3, stride=2, padding=1, expansion=2),
            InvertedResBlockMobile(
                24, 24, kernel_size=5, padding=2),
            InvertedResBlockMobile(
                24, 24, kernel_size=3, padding=1, expansion=2),
        )

        self.resblock2 = nn.Sequential(
            InvertedResBlockMobile(
                24, 32, kernel_size=3, stride=2, padding=1, expansion=2),
            InvertedResBlockMobile(
                32, 32, kernel_size=5, padding=2),
            InvertedResBlockMobile(
                32, 128, kernel_size=3, padding=1, expansion=4),
        )

        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.Conv2d(
                128, 24, kernel_size=1, padding=0, groups=1, bias=True),
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(24, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.Conv2d(24, 16, kernel_size=1, padding=0, groups=1, bias=True),
        )

        self.detector = nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=True)

        self.descriptor = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.Conv2d(
                128, 128, kernel_size=1, padding=0, groups=1, bias=True),
        )

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

