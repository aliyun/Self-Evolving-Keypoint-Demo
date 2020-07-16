
import argparse
import cv2
import numpy as np
import imageio
import os
import shutil
import torch
from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument('--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)')
parser.add_argument('--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model')
parser.add_argument('--max_edge', type=int, default=1600,
    help='maximum image size at network input')
parser.add_argument('--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input')
parser.add_argument('--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module')
parser.set_defaults(use_relu=True)
parser.add_argument('--dataroot', type=str,
    default='data/hpatches-benchmark/data/hpatches-release/',
    help='HPatches data.')
parser.add_argument('--path_result', type=str,
    default='data/hpatches-benchmark/data/descriptors/d2_net',
    help='image preprocessing (caffe or torch)')

args = parser.parse_args()

def GetGroupPatches(img_path, patch_size = 65, output_size = 65):
    #print('get a group patches from an image ' + img_path)
    img_input = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, channel = img_input.shape
    assert(patch_size == width)
    num_patches = int(height / patch_size)
    img_patches = np.zeros([num_patches, channel, output_size, output_size])
    for i in range(num_patches):
        img_patches[i,:,:,:] = np.transpose(cv2.resize(img_input[int(i*patch_size):int((i+1)*patch_size),:],
                (output_size, output_size)), [2, 0, 1])
    mean = np.array([103.939, 116.779, 123.68])
    img_patches = img_patches - mean.reshape([1, 3, 1, 1])
    return img_patches

# Export descriptor on hpatches.
print('Export descriptor on hpatches.')
print(args)
# Create model.
model = D2Net(model_file=args.model_file, use_relu=args.use_relu, use_cuda=use_cuda)

# calculate the descriptors for each patch in hpatches dataset
# for each sequence in hpatches dataset
for seq_name in os.listdir(args.dataroot):
    seq_path = os.path.join(args.dataroot, seq_name)
    seq_csv_path = os.path.join(args.path_result, seq_name)
    if os.path.exists(seq_csv_path):
        shutil.rmtree(seq_csv_path)
    os.makedirs(seq_csv_path)
    for img_name in os.listdir(seq_path):
        if img_name[-4:] != '.png':
            continue
        img_path = os.path.join(seq_path, img_name)
        patches = GetGroupPatches(img_path)
        # compute and save the descriptors
        batch = torch.from_numpy(patches)
        if use_cuda:
            batch = batch.cuda().float()
        with torch.no_grad():
            desc = model.dense_feature_extraction.forward(batch)
            _, _, height, width = desc.size()
            desc = desc[:, :, int(height/2), int(width/2)]
        desc = desc.cpu().numpy()
        csv_path = os.path.join(seq_csv_path, img_name[:-3]+'csv')
        f = open(csv_path, 'w')
        for i_patch in range(desc.shape[0]):
            for i_dim in range(desc.shape[1]-1):
                f.write('{:>8.5f}'.format(desc[i_patch][i_dim]) + ', ')
            f.write(str(desc[i_patch][i_dim+1]) + '\n')
        f.close()

