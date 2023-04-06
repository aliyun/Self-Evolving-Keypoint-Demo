
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Sigmoid(),
        )
        return
    def forward(self, input):
        attention = self.attention(input)
        return attention

class SEKDMotionMotion(torch.nn.Module):
    def __init__(self):
        super(SEKDMotionMotion, self).__init__()

        self.attention = AttentionBlock(128)

        self.feature3 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.motion = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )
        return
    def forward(self, input):
        feature0, feature1, feature2, feature3 = input
        feature3 = self.feature3(feature3)
        feature = feature2 + feature3
        attention = self.attention(feature)
        feature = attention * feature
        motion = self.motion(feature)
        return motion

class SEKDMotionDescriptor(torch.nn.Module):
    def __init__(self):
        super(SEKDMotionDescriptor, self).__init__()

        self.attention = AttentionBlock(128)

        self.feature3 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.descriptor = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=True),
        )
        return
    def forward(self, input):
        feature0, feature1, feature2, feature3 = input
        feature3 = self.feature3(feature3)
        feature = feature2 + feature3
        attention = self.attention(feature)
        feature = attention * feature
        descriptor = self.descriptor(feature)
        return descriptor

class SEKDMotionDetector(torch.nn.Module):
    def __init__(self):
        super(SEKDMotionDetector, self).__init__()

        self.attention = AttentionBlock(32)

        self.feature1 = nn.Sequential(
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.feature2 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )

        self.feature3 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1, bias=True),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        self.detector = nn.Sequential(
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1, bias=True),
        )
        return
    def forward(self, input):
        feature0, feature1, feature2, feature3 = input
        feature1 = self.feature1(feature1)
        feature2 = self.feature2(feature2)
        feature3 = self.feature3(feature3)
        feature = feature0 + feature1 + feature2 + feature3
        attention = self.attention(feature)
        feature = attention * feature
        detector = self.detector(feature)
        return detector

class SEKDMotionBackbone(torch.nn.Module):
    def __init__(self):
        super(SEKDMotionBackbone, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True)
        self.resblock0 = nn.Sequential(
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
            ResBlock(32, 32, kernel_size=3, padding=1),
        )

        self.resblock1 = nn.Sequential(
            ResBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
            ResBlock(64, 64, kernel_size=3, padding=1),
        )

        self.resblock2 = nn.Sequential(
            ResBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
        )

        self.resblock3 = nn.Sequential(
            ResBlock(128, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
            ResBlock(128, 128, kernel_size=3, padding=1),
        )

        return

    def forward(self, input):
        feature0 = self.resblock0(self.conv0(input))
        feature1 = self.resblock1(feature0)
        feature2 = self.resblock2(feature1)
        feature3 = self.resblock3(feature2)

        return feature0, feature1, feature2, feature3

class SEKDMotion(torch.nn.Module):
    """SEDKMotion model definition,
       simultaneously detect and describe keypoints.
    """
    def __init__(self):
        super(SEKDMotion, self).__init__()

        self.backbone = SEKDMotionBackbone()

        self.detector = SEKDMotionDetector()

        self.descriptor = SEKDMotionDescriptor()

        self.motion = SEKDMotionMotion()

        return

    def forward(self, input):
        features = self.backbone(input)

        score = self.detector(features)
        descriptor = self.descriptor(features)
        motion = self.motion(features)

        return score, descriptor, features[0], motion

