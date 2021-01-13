# Copyright (c) Alibaba Inc. All rights reserved.

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_displacement(response_map, xs, ys, radius = 1):
    assert(xs.shape == ys.shape)
    height, width = response_map.shape
    num_points = xs.shape[0]
    xs_offset = np.zeros(xs.shape)
    ys_offset = np.zeros(ys.shape)
    local_index_mask = np.zeros([2 * radius + 1, 2 * radius + 1, 2])
    for i in range(2 * radius + 1):
        local_index_mask[i, :, 0] = i - radius
    for j in range(2 * radius + 1):
        local_index_mask[:, j, 1] = j - radius
    for i_point in range(num_points):
        x = xs[i_point]
        y = ys[i_point]
        if x < radius or x >= height - radius:
            continue
        if y < radius or y >= width - radius:
            continue
        local_weight = response_map[x-radius:x+radius+1, y-radius:y+radius+1].copy()
        x_offset = - (local_weight[2,1] - local_weight[0, 1]) / (
            2.0 * (local_weight[0, 1] - 2.0 * local_weight[1,1] + local_weight[2,1]))
        y_offset = - (local_weight[1,2] - local_weight[1, 0]) / (
            2.0 * (local_weight[1, 0] - 2.0 * local_weight[1,1] + local_weight[1,2]))
        xs_offset[i_point] = min(max(x_offset, -radius), radius)
        ys_offset[i_point] = min(max(y_offset, -radius), radius)
    return xs_offset, ys_offset

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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True, padding_mode=padding_mode),
        )
        self.conv_residual = None
        if stride != 1 or in_channels != out_channels:
            self.conv_residual = nn.Conv2d(in_channels, out_channels,
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

class SEKDNet(torch.nn.Module):
    """SEDK network definition, simultaneously detect and describe keypoints.
    """
    def __init__(self):
        super(SEKDNet, self).__init__()
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

        self.deconv0 = nn.Sequential(
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                groups=8, output_padding=1, bias=True),
        )

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                groups=8, output_padding=1, bias=True),
        )

        self.detector = nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=True)

        self.descriptor = ResBlock(128, 128, kernel_size=3, padding=1)

        return

    def forward(self, input):
        _, _, height, width = input.shape
        feature0_ = self.resblock0(self.conv0(input))
        feature1_ = self.resblock1(feature0_)
        feature2 = self.resblock2(feature1_)

        feature1 = feature1_ + self.deconv0(feature2)
        feature0 = feature0_ + self.deconv1(feature1)

        location = self.detector(feature0)
        descriptor = self.descriptor(feature2)

        location = torch.nn.functional.softmax(location, dim = 1)
        location = location[:,1:,:,:]

        return location, descriptor

class SEKD(object):
    """ Use SEKD to detect keypoints and compute their descriptors. """
    def __init__(self, weights_path, confidence_threshold = 0.55,
                 nms_radius = 4, max_keypoints = 500, cuda = False,
                 multi_scale = False, sub_pixel_location = False):
        self.confidence_threshold = confidence_threshold
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.cuda = cuda
        self.multi_scale = multi_scale
        self.sub_pixel_location = sub_pixel_location

        # Load the network in inference mode.
        self.net = SEKDNet()
        self.net.load_state_dict(torch.load(weights_path)['state_dict'])
        self.net.eval()
        if self.cuda:
            self.net.cuda()

    def detectAndCompute(self, img, mask = None):
        """ For an input image, detect keypoints and compute their descriptors.
        Input
          img - HxW or HxWx3 numpy float32 image in range [0,1].
        Output
          keypoints - 3xN array with keypoint [col_i, row_i, confidence_i]^T.
          descriptors - 128xN array of unit normalized descriptors.
          """
        # TODO: discard the keypoints in mask.
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape

        keypoints, descriptors = self.detectAndComputeOneScale(img, mask = None)
        if self.multi_scale:
            min_size = 128
            down_scale_factor = 2.
            scale_factor = down_scale_factor
            height_current = int(height / scale_factor)
            width_current = int(width / scale_factor)
            while (height_current > min_size and width_current > min_size):
                #print(height_current)
                #print(width_current)
                img_current = cv2.resize(img, (width_current, height_current))
                img_current = img_current[0:int(height_current/4)*4, 0:int(width_current/4)*4]
                keypoints_current, descriptors_current = self.detectAndComputeOneScale(img_current)
                keypoints_current[0,:] = (keypoints_current[0,:] + 0.5) * scale_factor - 0.5
                keypoints_current[1,:] = (keypoints_current[1,:] + 0.5) * scale_factor - 0.5
                #print(keypoints_current.shape)
                keypoints = np.concatenate((keypoints, keypoints_current), axis=1)
                descriptors = np.concatenate((descriptors, descriptors_current), axis=1)
                scale_factor = scale_factor * down_scale_factor
                height_current = int(height / scale_factor)
                width_current = int(width / scale_factor)

        keypoints = keypoints
        scores = keypoints[2,:]
        keypoints = keypoints.T
        descriptors = descriptors.T
        inds = np.argsort(scores)[::-1]
        inds = inds[:self.max_keypoints]
        scores = scores[inds]
        keypoints = keypoints[inds, :].T
        descriptors = descriptors[inds, :].T

        return keypoints, descriptors

    def detectAndComputeOneScale(self, img, mask = None):
        """ For an input image, detect keypoints and compute their descriptors.
        Input
          img - HxW numpy float32 image in range [0,1].
        Output
          keypoints - 3xN array with keypoint [col_i, row_i, confidence_i]^T.
          descriptors - 128xN array of unit normalized descriptors.
          """
        height, width = img.shape
        img = img[0:int(height / 4) * 4, 0:int(width / 4) * 4]
        height, width = img.shape

        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            mask = torch.from_numpy(mask)
        if self.cuda:
            img_tensor = img_tensor.cuda()
            if mask is not None:
                mask = mask.cuda()
        # Calculate the results.
        with torch.no_grad():
            outs = self.net.forward(img_tensor)
            location, descriptors = outs[0], outs[1]
            location = location

            # NMS using max pooling.
            location_pooling = torch.nn.functional.max_pool2d(
                location, kernel_size = self.nms_radius * 2 + 1, stride = 1,
                padding = self.nms_radius)
            location = location.squeeze()
            location_pooling = location_pooling.squeeze()
            if mask is not None:
                xs, ys = torch.where((location > self.confidence_threshold) & (
                    location == location_pooling) & (mask > 0))
            else:
                xs, ys = torch.where((location > self.confidence_threshold) & (
                    location == location_pooling))

            # No keypoints detected.
            if xs.shape[0] == 0:
                return np.zeros([3, 0]), np.zeros([128, 0])

            # Sort keypoints.
            indicates = location[xs, ys].sort(descending=True)[1]
            if indicates.shape[0] > self.max_keypoints:
                indicates = indicates[:self.max_keypoints]
            xs = xs[indicates]
            ys = ys[indicates]

            # Refine keypoints
            if self.sub_pixel_location:
                xs_offset, ys_offset = calculate_displacement(
                    location.cpu().numpy(), xs.cpu().numpy(), ys.cpu().numpy())
                if self.cuda:
                    xs_offset = torch.from_numpy(xs_offset).cuda()
                    ys_offset = torch.from_numpy(ys_offset).cuda()
                xs = xs + xs_offset
                ys = ys + ys_offset

            keypoints = torch.cat((ys.to(location.dtype), xs.to(location.dtype),
                location[xs.to(torch.long), ys.to(torch.long)])).reshape([3, -1])
            keypoints = keypoints.detach().cpu().numpy()

            # Calculate descriptors via round the location.
            # descriptors = torch.nn.functional.interpolate(descriptors,
            #     size=[height, width], mode='bilinear')
            # descriptors = descriptors[0, :,
            #     keypoints[1,:].round().astype(np.int),
            #     keypoints[0,:].round().astype(np.int)]
            # descriptors = torch.nn.functional.normalize(descriptors, p = 2,
            #     dim = 0)

            # Calculate descriptors via interpolation.
            samp_pts = torch.cat((ys, xs)).reshape([2,-1])
            samp_pts = samp_pts.float()
            samp_pts[0, :] = 2. * (samp_pts[0, :]) / float(width) - 1.
            samp_pts[1, :] = 2. * (samp_pts[1, :]) / float(height) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous().view(1, 1, -1, 2)
            descriptors = torch.nn.functional.grid_sample(descriptors, samp_pts)
            descriptors = torch.nn.functional.normalize(descriptors).reshape([128, -1])

            descriptors = descriptors.detach().cpu().numpy()

        return keypoints, descriptors

