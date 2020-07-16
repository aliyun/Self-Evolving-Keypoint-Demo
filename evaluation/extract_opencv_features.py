# Copyright (c) Alibaba Inc. All rights reserved.

import argparse
import cv2
import numpy as np
import os
import shutil

def extract_opencv_features(args, method_name):
    print('Export {0} local features.'.format(method_name))
    # Calculate the detection result for each image in hpatches dataset.
    # For each sequence in hpatches dataset.
    if method_name == 'kaze':
        feature_extractor = cv2.KAZE_create()
    elif method_name == 'sift':
        feature_extractor = cv2.xfeatures2d.SIFT_create()
    elif method_name == 'surf':
        feature_extractor = cv2.xfeatures2d.SURF_create()
    elif method_name == 'akaze':
        feature_extractor = cv2.AKAZE_create()
    elif method_name == 'brisk':
        feature_extractor = cv2.BRISK_create()
    elif method_name == 'orb':
        feature_extractor = cv2.ORB_create()
    else:
        print('Unknown method: ' + method_name)
        return
    for seq_name in sorted(os.listdir(args.images_dir)):
        seq_path = os.path.join(args.images_dir, seq_name)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.ppm':
                continue
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape
            keypoints_list, descriptors = feature_extractor.detectAndCompute(img, None)

            keypoints = np.zeros([len(keypoints_list), 2])
            scores = np.zeros(len(keypoints_list))

            for i in range(len(keypoints_list)):
                keypoints[i, 0] = keypoints_list[i].pt[0]
                keypoints[i, 1] = keypoints_list[i].pt[1]
                scores[i] = keypoints_list[i].response

            inds = np.argsort(scores)
            keypoints = keypoints[inds[:-501:-1], :]
            scores = scores[inds[:-501:-1]]

            descriptors = descriptors[inds[:-501:-1], :]
            if descriptors.dtype == np.uint8:
                dim_descriptor = descriptors.shape[1] * 8
                descriptors_float = np.zeros([keypoints.shape[0], dim_descriptor])
                for i in range(keypoints.shape[0]):
                    for j in range(dim_descriptor):
                        descriptors_float[i][j] = bool(descriptors[i][int(j/8)] & (1 << j%8))
                descriptors = descriptors_float


            descriptors = descriptors.astype(np.float)
            det_dir = os.path.join(args.features_dir, seq_name)
            if not os.path.isdir(det_dir):
                os.makedirs(det_dir)
            det_path = os.path.join(det_dir, img_name + '.' + method_name)
            print(det_path)
            with open(det_path, 'wb') as output_file:
                np.savez(output_file, keypoints=keypoints, scores=scores, descriptors=descriptors)

def extract_opencv_desc(args, method_name):
    # Calculate the descriptor for each patch in hpatches dataset.
    # Current only support surf and brisk.
    # Results of sift and orb are directly downloaded from HPatches.
    # As the unknown parameters of akaze, kaze keypoints, we cannot compute
    # their descriptors.
    print('Export {0} descriptors.'.format(method_name))
    PATCH_WIDTH = 65
    if method_name == 'surf':
        feature_extractor = cv2.xfeatures2d.SURF_create()
        keypoints_list = [cv2.KeyPoint(x=(PATCH_WIDTH/2), y=(PATCH_WIDTH/2),
            _size=(PATCH_WIDTH/2), _angle=0., _response=1., _octave=0, _class_id=0)]
    elif method_name == 'brisk':
        feature_extractor = cv2.BRISK_create()
        keypoints_list = [cv2.KeyPoint(x=(PATCH_WIDTH/2), y=(PATCH_WIDTH/2),
            _size=(PATCH_WIDTH/5), _angle=0., _response=1., _octave=0, _class_id=0)]
    else:
        print('Unknown method: ' + method_name)
        return

    # Calculate the descriptor for each patch in hpatches dataset.
    result_dir = os.path.join(args.desc_dir, method_name)
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
            desc_path = os.path.join(seq_csv_path, img_name[:-4] + '.csv')
            print(desc_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            patches = img.reshape([-1, PATCH_WIDTH, PATCH_WIDTH])
            num_patches = patches.shape[0]
            output_file = open(desc_path, 'w')
            for i_patch in range(num_patches):
                patch = patches[i_patch, :, :]
                _, descriptors = feature_extractor.compute(patch, keypoints_list)
                desc = descriptors[0, :]
                if desc.dtype == np.uint8:
                    for i_dim in range(desc.shape[0]*8-1):
                        output_file.write('{0}'.format(int(bool(desc[int(i_dim/8)] & (1 << i_dim%8)))) + ', ')
                    output_file.write(str(int(bool(desc[-1] & (1 << (i_dim+1)%8)))) + '\n')
                else:
                    for i_dim in range(desc.shape[0]-1):
                        output_file.write('{:>8.5f}'.format(desc[i_dim]) + ', ')
                    output_file.write(str(desc[-1]) + '\n')
            output_file.close()

