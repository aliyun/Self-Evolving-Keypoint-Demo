
import cv2
import numpy as np
import datetime
from torch.utils.data import *
import torch
from collections import OrderedDict
import os

from demo_superpoint import *

def GetGroupPatches(img_path, patch_size = 65, output_size = 33):
    print('get a group patches from an image ' + img_path)
    img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img_input.shape
    assert(patch_size == width)
    num_patches = int(height / patch_size)
    img_patches = np.zeros([num_patches, 1, output_size, output_size])
    for i in range(num_patches):
        img_patches[i,0,:,:] = cv2.resize(img_input[int(i*patch_size):int((i+1)*patch_size),:],
                (output_size, output_size))
    return img_patches

def LoadSuperpointModel(weights_path):
    print('load model from ' + weights_path)
    cuda = True
    # prepare net
    net = SuperPointNet()
    if cuda:
        # Train on GPU, deploy on GPU.
        net.load_state_dict(torch.load(weights_path))
        net = net.cuda()
    else:
        # Train on GPU, deploy on CPU.
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

    return net

def ComputeHpatchesDescriptor(weights_path, desc_path, hpatches_path):
    print('compute hpatches descriptor')
    # load model
    net = LoadSuperpointModel(weights_path)
    # for each sequence in hpatches dataset
    for seq_name in os.listdir(hpatches_path):
        seq_path = os.path.join(hpatches_path, seq_name)
        seq_csv_path = os.path.join(desc_path, seq_name)
        os.mkdir(seq_csv_path)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.png':
                continue
            img_path = os.path.join(seq_path, img_name)
            patches = GetGroupPatches(img_path)
            # compute and save the descriptors
            batch = torch.from_numpy(patches)
            batch = batch.cuda().float()
            batch = batch / 255
            with torch.no_grad():
                semi, desc = net.forward(batch)
            desc = desc.cpu().numpy()
            csv_path = os.path.join(seq_csv_path, img_name[:-3]+'csv')
            f = open(csv_path, 'w')
            x = int(desc.shape[2] / 2)
            y = int(desc.shape[3] / 2)
            for i_patch in range(desc.shape[0]):
                for i_dim in range(desc.shape[1]-1):
                    f.write('{:>8.5f}'.format(desc[i_patch][i_dim][x][y]) + ', ')
                f.write(str(desc[i_patch][i_dim+1][x][y]) + '\n')
            f.close()

if __name__ == '__main__':
    print('main')
    weights_path = './superpoint_v1.pth'
    desc_path = './data/hpatches_desc/superpoint/'
    hpatches_path = 'path_to/hpatches-release/'
    ComputeHpatchesDescriptor(weights_path, desc_path, hpatches_path)

