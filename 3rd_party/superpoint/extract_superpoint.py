
import argparse
import cv2
import datetime
import math
import numpy as np
import os
import PIL
import random
from scipy.spatial.distance import cdist
import shutil
import sys

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from demo_superpoint import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SEKD')
# Model options
parser.add_argument('--dataroot', type=str,
                    default='/home/songyoff/projects/data/hpatches/hpatches_benchmark_homo/hpatches_seq_resize/',
                    help='Path of the dataset.')
parser.add_argument('--model_path', type=str,
                    default='superpoint_v1.pth',
                    help='Path of the model.')
# Device options
parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Enable using CUDA for acceleration.')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='GPU id(s) used by the cuda.')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()

args.confidence_threshold = 0.001
args.maxinum_points = 500
args.nms_radius = 4
args.refine_radius = 4
args.detector_cell = 8

args.use_cuda = args.use_cuda and torch.cuda.is_available()
if args.use_cuda:
    gpu_ids = [int(index) for index in args.gpu_ids.replace(',', ' ').split()]
    args.gpu_ids = gpu_ids

print (("NOT " if not args.use_cuda else "") + "Using cuda")

if args.use_cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# set random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def calculate_displacement(response_map, xs, ys, radius = 4):
    assert(xs.shape == ys.shape)
    num_points = xs.shape[0]
    xs_offset = np.zeros(xs.shape)
    ys_offset = np.zeros(ys.shape)
    #return xs_offset, ys_offset
    local_index_mask = np.zeros([2*radius + 1, 2*radius+1, 2])
    for i in range(2*radius + 1):
        for j in range(2*radius + 1):
            local_index_mask[i, j, 0] = i - radius
            local_index_mask[i, j, 1] = j - radius
    for i_point in range(num_points):
        x = xs[i_point]
        y = ys[i_point]
        local_weight = response_map[x-radius:x+radius+1, y-radius:y+radius+1].copy()
        #local_weight = np.exp(local_weight)
        #local_weight = local_weight / np.sum(local_weight)
        local_weight = 1.3 * local_weight / local_weight[radius, radius]
        x_offset = np.sum(local_index_mask[:,:,0] * local_weight)
        y_offset = np.sum(local_index_mask[:,:,1] * local_weight)
        xs_offset[i_point] = min(max(x_offset, radius), radius)
        ys_offset[i_point] = min(max(y_offset, radius), radius)
        xs_offset[i_point] = x_offset
        ys_offset[i_point] = y_offset
        #if i_point == 150:
        #    print('x_offset: {0}, y_offset: {1}'.format(x_offset, y_offset))
    return xs_offset, ys_offset

# use non maximax suppress (nms) to find robust keypoints
def non_maximax_suppress_refine(response_map_init, confidence_threshold = 0.2, maxinum_points = 500, radius = 4):
    response_map = response_map_init.copy()
    height, width = response_map.shape
    response_map[0:2*radius, :] = 0
    response_map[:, 0:2*radius] = 0
    response_map[height-2*radius:height, :] = 0
    response_map[:, width-2*radius:width] = 0
    xs, ys = np.where(response_map > confidence_threshold) # Confidence threshold.
    in_corners = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    in_corners[0, :] = xs
    in_corners[1, :] = ys
    in_corners[2, :] = response_map[xs, ys]

    grid = np.zeros((height, width)).astype(np.int) # Track NMS data.
    inds_map_to_point = np.zeros((height, width)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds_sort = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds_sort]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return grid
    if rcorners.shape[1] == 1:
        grid[rcorners[0,0], rcorners[1,0]] = 1
        return grid
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[0,i], rcorners[1,i]] = -1
        inds_map_to_point[rc[0], rc[1]] = i
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        if grid[rc[0], rc[1]] == -1: # If not yet suppressed.
            #if (np.max(response_map[rc[0]-radius:rc[0]+radius+1, rc[1]-radius:rc[1]+radius+1]) ==
            #    response_map[rc[0], rc[1]]):
            grid[rc[0]-radius:rc[0]+radius+1, rc[1]-radius:rc[1]+radius+1] = 0
            grid[rc[0], rc[1]] = 1
            count += 1
            #else:
            #    grid[rc[0], rc[1]] = 0
            if count == maxinum_points:
                break
    grid[grid < 0] = 0

    xs, ys = np.where(grid == 1)
    #x_offset, y_offset = calculate_displacement(response_map, xs_init, ys_init, radius = args.refine_radius)
    #xs = xs + x_offset
    #ys = ys + y_offset
    xs = xs.clip(0, height-1)
    ys = ys.clip(0, width-1)
    coord = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
    return coord

def export_detection_on_hpatches():
    print('Export detection results on hpatches.')
    model = SuperPointNet()
    model_desc_dict = torch.load(args.model_path)
    model.load_state_dict(model_desc_dict, strict=False)
    if args.use_cuda:
        model.cuda(args.gpu_ids[0])
    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    for seq_name in os.listdir(args.dataroot):
        seq_path = os.path.join(args.dataroot, seq_name)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.ppm':
                continue
            torch.cuda.empty_cache()
            img_path = os.path.join(seq_path, img_name)
            det_path = os.path.join(seq_path, img_name + '.superpoint')
            print(det_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape
            img_height = int(img_height / 8) * 8
            img_width = int(img_width / 8) * 8
            img = img[0:img_height, 0:img_width]
            img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if args.use_cuda:
                img = img.cuda(args.gpu_ids[0]).float()
            img = img / 255.
            with torch.no_grad():
                location, descriptor = model.forward(img)
                torch.cuda.empty_cache()
                #location = torch.nn.functional.interpolate(location, size=[img_height, img_width], mode='bilinear')
                descriptor = torch.nn.functional.interpolate(descriptor, size=[img_height, img_width], mode='bilinear')
                location = F.softmax(location, dim = 1)

            location = location[0,:-1,:,:].detach().squeeze().cpu().numpy()
            _, height_det_out, width_det_out = location.shape
            location = location.transpose(1, 2, 0)
            location = location.reshape([height_det_out, width_det_out, args.detector_cell, args.detector_cell])
            location = np.transpose(location, [0, 2, 1, 3])
            location = np.reshape(location, [height_det_out*args.detector_cell, width_det_out*args.detector_cell])

            coord = non_maximax_suppress_refine(location, confidence_threshold = args.confidence_threshold,
                    maxinum_points = args.maxinum_points, radius = args.nms_radius)
            keypoints = coord.transpose()[:,::-1]
            print(keypoints.shape)
            #assert(keypoints.shape[0] > 2000)

            scores = location[coord[0,:].round().astype(np.int), coord[1,:].round().astype(np.int)]
            print(scores.shape)

            descriptor = descriptor[0,:,coord[0,:].round().astype(np.int), coord[1,:].round().astype(np.int)]
            descriptor = F.normalize(descriptor, p=2, dim = 0)
            descriptors = descriptor.transpose(0, 1).detach().cpu().numpy()
            print(descriptors.shape)

            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores, descriptors=descriptors)
            #cv2.imwrite(det_path[:-10] + 'prob_' + det_path[-10:] + '.png',
            #        (location-location.min())/(location.max()-location.min())*128.)

if __name__ == '__main__':
    export_detection_on_hpatches()

