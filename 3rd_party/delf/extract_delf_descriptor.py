
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import sys
import shutil
import time

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf import extractor

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

def _ReadImageList(list_path):
    """Helper function to read image paths.
    Args: list_path: Path to list of images, one image path per line.
    Returns: image_paths: List of image paths.
    """
    with tf.gfile.GFile(list_path, 'r') as f:
        image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

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

def main(unused_argv):
    print('Export descriptor on hpatches.')
    print(cmd_args)
    tf.logging.set_verbosity(tf.logging.INFO)
    # Read list of images.
    tf.logging.info('Reading list of images...')
    image_paths = _ReadImageList(cmd_args.list_images_path)
    num_images = len(image_paths)
    tf.logging.info('done! Found %d images', num_images)
    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
        text_format.Merge(f.read(), config)

    # Create output directory if necessary.
    if os.path.isdir(cmd_args.output_dir) == False:
        os.mkdir(cmd_args.output_dir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Reading list of images.
        filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image_tf = tf.image.decode_png(value, channels=3)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            extractor_fn = extractor.MakeExtractor(sess, config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()
            for i in range(num_images):
                # Write to log-info once in a while.
                if i == 0:
                    tf.logging.info('Starting to extract DELF features from images...')
                    # Output node names to file.
                    #node_names = [node.name for node in tf.get_default_graph().as_graph_def().node]
                    #f = open('node_names.txt', 'w')
                    #for name in node_names:
                    #    f.write(name + '\n')
                    #f.close()
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = (time.clock() - start)
                    tf.logging.info(
                        'Processing image %d out of %d, last %d '
                        'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
                        elapsed)
                    start = time.clock()

                # Get next image.
                patches = sess.run(image_tf)
                patches = patches.reshape((-1, 65, 65, 3))
                num_patches = patches.shape[0]

                input_dir, img_name = os.path.split(image_paths[i])
                out_desc_dir = os.path.join(cmd_args.output_dir, input_dir.split('/')[-1])
                out_desc_fullpath = os.path.join(out_desc_dir, img_name[:-3]+'csv')
                if os.path.isdir(out_desc_dir) == False:
                    os.mkdir(out_desc_dir)
                print(out_desc_fullpath)
                output_file = open(out_desc_fullpath, 'w')
                for i_patch in range(num_patches):
                    # Extract and save features.
                    im = patches[i_patch, :, :, :]
                    (locations_out, descriptors_out, feature_scales_out,
                        attention_out, feature_map_out) = extractor_fn(im)

                    # Output descriptors to file.
                    desc = feature_map_out[0, int(feature_map_out.shape[1]/2), int(feature_map_out.shape[2]/2), :]
                    for i_dim in range(desc.shape[0]-1):
                        output_file.write('{:>8.5f}'.format(desc[i_dim]) + ', ')
                    output_file.write(str(desc[i_dim+1]) + '\n')

                output_file.close()

            # Finalize enqueue threads.
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument(
            '--config_path', type=str,
            default='delf_config_hpatches.pbtxt',
            help=""" Path to DelfConfig proto text file with configuration to be used for DELF extraction.  """)
    parser.add_argument(
            '--list_images_path', type=str,
            #default='hpatches_descriptor_img_paths.txt',
            default='hpatches_descriptor_img_paths.txt',
            help=""" Path to list of images whose DELF features will be extracted.  """)
    parser.add_argument(
            '--output_dir', type=str,
            default='/home/songyoff/projects/data/hpatches/hpatches-benchmark/data/descriptors/delf/',
            help=""" Directory where DELF features will be written to. Each image's features
            will be written to a file with same name, and extension replaced by .delf.  """)

    cmd_args, unparsed = parser.parse_known_args()

    app.run(main=main, argv=[sys.argv[0]] + unparsed)

