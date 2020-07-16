# Copyright (c) Alibaba Inc. All rights reserved.

from nets import SEKD, SEKDNet
from utils import GetGroupPatches

import argparse
import cv2
import numpy as np
import os
import shutil
import torch

def extract_sekd(args):
    print('Export SEKD local features.')
    feature_extractor = SEKD(weights_path=args.weights_path,
        confidence_threshold = args.conf_thresh, nms_radius = args.nms_radius,
        max_keypoints = args.max_keypoints, cuda = args.cuda)

    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    for seq_name in sorted(os.listdir(args.images_dir)):
        seq_path = os.path.join(args.images_dir, seq_name)
        for img_name in sorted(os.listdir(seq_path)):
            if img_name[-4:] != '.ppm':
                continue
            if args.cuda:
                torch.cuda.empty_cache()
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0

            keypoints, descriptors = feature_extractor.detectAndCompute(img)

            keypoints = keypoints
            scores = keypoints[2,:]
            keypoints = keypoints[0:2].T
            descriptors = descriptors.T

            det_dir = os.path.join(args.features_dir, seq_name)
            if not os.path.isdir(det_dir):
                os.makedirs(det_dir)
            det_path = os.path.join(det_dir, img_name + '.sekd')
            print(det_path)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores,
                    descriptors=descriptors)

def extract_sekd_desc(args):
    print('Export SEKD local features.')
    model = SEKDNet()
    model_dict = torch.load(args.weights_path)
    model.load_state_dict(model_dict['state_dict'], strict=True)
    model.eval()
    if args.cuda:
        model.cuda()

    # Calculate the descriptor for each patch in hpatches dataset.
    result_dir = os.path.join(args.desc_dir, 'sekd')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    for seq_name in sorted(os.listdir(args.patches_dir)):
        seq_path = os.path.join(args.patches_dir, seq_name)
        seq_csv_path = os.path.join(result_dir, seq_name)
        if os.path.exists(seq_csv_path):
            shutil.rmtree(seq_csv_path)
        os.makedirs(seq_csv_path)
        for img_name in sorted(os.listdir(seq_path)):
            if img_name[-4:] != '.png':
                continue
            img_path = os.path.join(seq_path, img_name)
            patches = GetGroupPatches(img_path, output_size = 32)
            patches = patches.astype(np.float) / 255.
            # compute and save the descriptors
            with torch.no_grad():
                batch = torch.from_numpy(patches).to(torch.float32)
                if args.cuda:
                    torch.cuda.empty_cache()
                    batch = batch.cuda()
                _, desc = model.forward(batch)
                desc = torch.nn.functional.normalize(desc, dim=1)
                _, _, height, width = desc.size()
                desc = desc[:, :, int(height/2), int(width/2)]
                desc = desc.detach().cpu().numpy()
                csv_path = os.path.join(seq_csv_path, img_name[:-3]+'csv')
                print(csv_path)
                f = open(csv_path, 'w')
                for i_patch in range(desc.shape[0]):
                    for i_dim in range(desc.shape[1]-1):
                        f.write('{:>8.5f}'.format(desc[i_patch][i_dim]) + ', ')
                    f.write(str(desc[i_patch][i_dim+1]) + '\n')
                f.close()

