# Copyright (c) Alibaba Inc. All rights reserved.

from nets import SEKD
import evaluation

import argparse
import cv2
import numpy as np
import os
import torch

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Resize HPatches sequence images.')
parser.add_argument('--images_dir', type=str,
    default='./data/hpatches-dataset/hpatches-sequences-resize',
    help='Dir of the hpatches-sequences-images.')
parser.add_argument('--features_dir', type=str,
    default='./data/hpatches-dataset/features',
    help='Dir to store the local features of hpatches-sequences-images.')
parser.add_argument('--patches_dir', type=str,
    default='./data/hpatches-benchmark/data/hpatches-release',
    help='Dir of the hpatches data.')
parser.add_argument('--desc_dir', type=str,
    default='./data/hpatches-benchmark/data/descriptors',
    help='Dir to store the descriptors of each patch in hpatches.')

parser.add_argument('--weights_path', type=str, default='./assets/sekd.pth',
    help='Path to pretrained weights file (default: sekd.pth).')
parser.add_argument('--max_keypoints', type=int, default=500,
    help='Maximum keypoints of detection results (default: 500).')
parser.add_argument('--nms_radius', type=int, default=4,
    help='Non Maximum Suppression (NMS) radius (default: 4).')
parser.add_argument('--conf_thresh', type=float, default=0.4,
    help='Detector confidence threshold (default: 0.4).')
parser.add_argument('--cuda', action='store_true',
    help='Use cuda to speed up network processing speed (default: False)')

if __name__ == '__main__':
    args = parser.parse_args()
    evaluation.extract_sekd(args)
    evaluation.extract_opencv_features(args, 'kaze')
    evaluation.extract_opencv_features(args, 'sift')
    evaluation.extract_opencv_features(args, 'surf')
    evaluation.extract_opencv_features(args, 'akaze')
    evaluation.extract_opencv_features(args, 'brisk')
    evaluation.extract_opencv_features(args, 'orb')

    evaluation.extract_sekd_desc(args)
    evaluation.extract_opencv_desc(args, 'surf')
    evaluation.extract_opencv_desc(args, 'brisk')

