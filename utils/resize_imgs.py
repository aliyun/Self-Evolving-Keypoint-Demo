# Copyright (c) Alibaba Inc. All rights reserved.

import argparse
import cv2
import numpy as np
import os
import shutil

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Resize HPatches sequence images.')
parser.add_argument('--input_dir', type=str, default='./hpatches-sequences-release',
    help='Dir of the hpatches-sequences-release.')
parser.add_argument('--output_dir', type=str, default='./hpatches-sequences-resize',
    help='Dir to store the resized result of hpatches-sequences-release.')
parser.add_argument('--max_edge', type=int, default=640,
    help='Max edge of height or width, resize the image if it is too large.')

def resize_imgs(input_dir, output_dir, max_edge):
    cell = 16
    print('Resize all images in {0} and save into {1}.'.format(input_dir, output_dir))
    min_height = 1e5
    min_width = 1e5
    for seq_name in os.listdir(input_dir):
        seq_path = os.path.join(output_dir, seq_name)
        if os.path.exists(seq_path):
            shutil.rmtree(seq_path)
        os.makedirs(seq_path)
        print(seq_path)

        scale = 1.
        for i in range(1,7):
            img_path = os.path.join(input_dir, seq_name, str(i) + '.ppm')
            result_path = os.path.join(output_dir, seq_name, str(i) + '.ppm')
            print('  Resize image {0} and save into {1}.'.format(img_path, result_path))
            img = cv2.imread(img_path)
            height_init, width_init, _ = img.shape
            if height_init < min_height:
                min_height = height_init
            if width_init < min_width:
                min_width = width_init
            scale0 = max_edge / height_init
            scale1 = max_edge / width_init
            scale = scale0 if scale0 < scale1 else scale1
            if scale < 1:
                img = cv2.resize(img, (int(scale*width_init), int(scale*height_init)), interpolation=cv2.INTER_LINEAR)
            height_out = int(img.shape[0]/cell)*cell
            width_out = int(img.shape[1]/cell)*cell
            img = img[:height_out, :width_out, :]
            cv2.imwrite(result_path, img)

        for i in range(2, 7):
            homo_path = os.path.join(input_dir, seq_name, 'H_1_' + str(i))
            result_path = os.path.join(output_dir, seq_name, 'H_1_' + str(i))
            homography = np.loadtxt(homo_path)
            homography[0,2] = homography[0,2] * scale
            homography[1,2] = homography[1,2] * scale
            homography[2,0] = homography[2,0] / scale
            homography[2,1] = homography[2,1] / scale
            homography = np.savetxt(result_path, homography)

if __name__ == '__main__':
    args = parser.parse_args()
    resize_imgs(args.input_dir, args.output_dir, args.max_edge)

