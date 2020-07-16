# ==============================================================================
"""Extracts DELF features from on hpatches, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
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

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def main(unused_argv):
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
  #if not tf.gfile.Exists(cmd_args.output_dir):
  #  tf.gfile.MakeDirs(cmd_args.output_dir)

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
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info(
              'Processing image %d out of %d, last %d '
              'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
              elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        out_desc_fullpath = image_paths[i][:-3] + 'ppm' + _DELF_EXT
        if tf.gfile.Exists(out_desc_fullpath):
          tf.logging.info('Skipping %s', image_paths[i])
          continue

        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = extractor_fn(im)

        with open(out_desc_fullpath, 'wb') as output_file:
            np.savez(output_file, keypoints=locations_out[:,::-1], scores=attention_out,
                  descriptors=descriptors_out)

        #feature_io.WriteToFile(out_desc_fullpath, locations_out,
        #                       feature_scales_out, descriptors_out,
        #                       attention_out)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_hpatches.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
