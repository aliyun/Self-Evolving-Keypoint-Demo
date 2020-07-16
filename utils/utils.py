# Copyright (c) Alibaba Inc. All rights reserved.

import cv2
import numpy as np

def GetGroupPatches(img_path, patch_size = 65, output_size = 65):
    #print('get a group patches from an image ' + img_path)
    img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img_input.shape
    channel = 1
    assert(patch_size == width)
    num_patches = int(height / patch_size)

    img_patches = cv2.resize(img_input, (output_size, output_size*num_patches))
    img_patches = img_patches.reshape(
        [num_patches, channel, output_size, output_size])
    return img_patches

