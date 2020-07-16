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


def build_networks(config, photo, is_training):

    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)

    # extract patches
    kpts = det_endpoints['kpts']
    print("\nkpts.name: " + kpts.name + "\n")
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]
    print("\ndesc_feats.name: " + desc_feats.name + "\n")

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }

    return ops

def main(config):

    # Build Networks
    tf.reset_default_graph()

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing

    ops = build_networks(config, photo_ph, is_training)

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

    # Ready to feed input images
    #img_paths = [x.path for x in os.scandir(config.in_dir) if x.name.endswith('.jpg') or x.name.endswith('.png')]
    #print('Found {} images...'.format(len(img_paths)))

    #if not os.path.exists(config.out_dir):
    #    os.makedirs(config.out_dir)

    avg_elapsed_time = 0

    for seq_name in os.listdir(config.in_dir):
        seq_path = os.path.join(config.in_dir, seq_name)
        for img_name in os.listdir(seq_path):
            if img_name[-4:] != '.ppm':
                continue
            img_path = os.path.join(seq_path, img_name)
            det_path = os.path.join(seq_path, img_name + '.lfnet')
            photo = imread(img_path)
            height, width = photo.shape[:2]
            longer_edge = max(height, width)
            if config.max_longer_edge > 0 and longer_edge > config.max_longer_edge:
                if height > width:
                    new_height = config.max_longer_edge
                    new_width = int(width * config.max_longer_edge / height)
                else:
                    new_height = int(height * config.max_longer_edge / width)
                    new_width = config.max_longer_edge
                photo = cv2.resize(photo, (new_width, new_height))
                height, width = photo.shape[:2]
            rgb = photo.copy()
            if photo.ndim == 3 and photo.shape[-1] == 3:
                photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
            photo = photo[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
            assert photo.ndim == 4 # [1,H,W,1]
            feed_dict = {
                photo_ph: photo,
            }
            print("\nphoto_ph.name: " + photo_ph.name + "\n")
            if config.full_output:
                fetch_dict = {
                    'kpts': ops['kpts'],
                    'feats': ops['feats'],
                    'kpts_scale': ops['kpts_scale'],
                    'kpts_ori': ops['kpts_ori'],
                    'scale_maps': ops['scale_maps'],
                    'degree_maps': ops['degree_maps'],
                }
                outs = sess.run(fetch_dict, feed_dict=feed_dict)
                # draw key
                kp_img = draw_keypoints(rgb, outs['kpts'])
                scale_range = config.net_max_scale-config.net_min_scale
                if scale_range == 0:
                    scale_range = 1.0
                scale_img = (outs['scale_maps'][0]*255/scale_range).astype(np.uint8)
                ori_img = (outs['degree_maps'][0]*255).astype(np.uint8)
                #print(outs['kpts'].shape)
                #print(outs['feats'].shape)
                #print(outs['kpts_scale'].shape)
                keypoints = outs['kpts']
                scores = outs['kpts_scale']
                descriptors = outs['feats']
                with open(det_path, 'wb') as output_file:
                    np.savez(output_file, keypoints=keypoints, scores=scores, descriptors=descriptors)
                #out_img_path = os.path.join(config.out_dir, os.path.basename(img_path))
                #imsave(out_img_path, kp_img)
                #imsave(out_img_path+'-scl.jpg', scale_img)
                #imsave(out_img_path+'-ori.jpg', ori_img)
                #np.savez(out_img_path+'.npz', kpts=outs['kpts'], descs=outs['feats'], size=np.array([height, width]),
                #         scales=outs['kpts_scale'], oris=outs['kpts_ori'])
    print('Done.')

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    io_arg = add_argument_group('In/Out', parser)
    io_arg.add_argument('--in_dir', type=str,
            default='/home/songyoff/projects/data/hpatches/hpatches_benchmark_homo/hpatches_seq_resize/',
            help='input image directory')
    # io_arg.add_argument('--in_dir', type=str, default='./release/outdoor_examples/images/sacre_coeur/dense/images',
    #                         help='input image directory')
    io_arg.add_argument('--out_dir', type=str, default='./dump_feats',
                            help='where to save keypoints')
    io_arg.add_argument('--full_output', type=str2bool, default=True,
                            help='dump keypoint image')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument('--model', type=str, default='./release/models/outdoor/',
                            help='model file or directory')
    model_arg.add_argument('--top_k', type=int, default=500,
                            help='number of keypoints')
    model_arg.add_argument('--max_longer_edge', type=int, default=640,
                            help='resize image (do nothing if max_longer_edge <= 0)')

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
