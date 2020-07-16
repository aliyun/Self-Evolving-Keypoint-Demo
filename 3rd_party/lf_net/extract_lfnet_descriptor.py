#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle

from mydatasets import *

from det_tools import *
from eval_tools import draw_keypoints
from common.tf_train_utils import get_optimizer
from imageio import imread, imsave
from inference import *

MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

def build_networks(config, patches, is_training):
    if config.input_inst_norm:
        print('Apply instance norm on input patches')
        patches1 = instance_normalization(patches)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(patches, reuse=False) # [B*K,D]
    print("\ndesc_feats.name: " + desc_feats.name + "\n")

    ops = {
        'patches': patches,
        'is_training': is_training,
        'desc': desc_feats,
        # EXTRA
        'desc_endpoints': desc_endpoints,
    }

    return ops

def main(config):
    # Build Networks
    tf.reset_default_graph()
    patches_ph = tf.placeholder(tf.float32, [None, 32, 32, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing
    ops = build_networks(config, patches_ph, is_training)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver()
    print('Load trained models...')
    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)

    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    
    print('Done.')

    print('Calculating descriptors ...')
    avg_elapsed_time = 0
    for seq_name in sorted(os.listdir(config.in_dir)):
        seq_path = os.path.join(config.in_dir, seq_name)
        desc_out_dir = os.path.join(config.out_dir, seq_name)
        if not os.path.isdir(desc_out_dir):
            os.makedirs(desc_out_dir)
        for img_name in sorted(os.listdir(seq_path)):
            if img_name[-4:] != '.png':
                continue
            img_path = os.path.join(seq_path, img_name)
            desc_path = os.path.join(desc_out_dir, img_name[:-4] + '.csv')
            print(desc_path)
            img = imread(img_path)
            height, width = img.shape[:2]
            num_patches = int(height/65)
            rgb = img.copy()
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (32, 32*num_patches), interpolation=cv2.INTER_LINEAR)
            img = img[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
            assert img.ndim == 4 # [1,H,W,1]
            patches = img.reshape([-1, 32, 32, 1])
            feed_dict = {
                patches_ph: patches,
            }
            #print("\npatches_ph.name: " + patches_ph.name + "\n")
            fetch_dict = {
                'desc': ops['desc'],
            }
            outs = sess.run(fetch_dict, feed_dict=feed_dict)
            desces = outs['desc']
            #print(desces.shape)
            #print(type(desces))
            output_file = open(desc_path, 'w')
            for i_patch in range(num_patches):
                desc = desces[i_patch, :]
                for i_dim in range(desc.shape[0]-1):
                    output_file.write('{:>8.5f}'.format(desc[i_dim]) + ', ')
                output_file.write('{:>8.5f}'.format(desc[-1]) + '\n')
            output_file.close()
            #break
        #break
    print('Done.')

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument(
            '--num_threads', type=int, default=8,
            help='the number of threads (for dataset)')

    io_arg = add_argument_group('In/Out', parser)
    io_arg.add_argument(
            '--in_dir', type=str, default='/home/songyoff/projects/data/hpatches/hpatches-release/',
            help='input image directory')
    io_arg.add_argument(
            '--out_dir', type=str, default='/home/songyoff/projects/data/hpatches/hpatches-benchmark/data/descriptors/lfnet',
            help='where to save keypoints')
    io_arg.add_argument(
            '--full_output', type=str2bool, default=True,
            help='dump keypoint image')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument(
            '--model', type=str, default='./release/models/outdoor/',
            help='model file or directory')
    model_arg.add_argument(
            '--top_k', type=int, default=500,
            help='number of keypoints')

    tmp_config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            setattr(config, attr, dst_val)

    print(config)
    main(config)

