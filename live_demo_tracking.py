# Copyright (c) Alibaba Inc. All rights reserved.

import nets
from utils import Tracker, Video

import argparse
import cv2
import numpy as np
import os
import torch

# Parse command line arguments.
parser = argparse.ArgumentParser(description='SEKD Demo.')
parser.add_argument('input', type=str, default='camera',
    help='Image directory or movie file or camera (default: camera).')
parser.add_argument('--camera_id', type=int, default=0,
    help='OpenCV webcam video capture ID (default: 0).')
parser.add_argument('--img_ext', type=str, default='.jpg',
    help='Glob match if directory of images is specified (default: *.jpg).')
parser.add_argument('--max_height', type=int, default=480,
    help='Max height of input image (default: 480).')
parser.add_argument('--max_width', type=int, default=640,
    help='Max width of input image (default: 640).')

parser.add_argument('--model_name', default='SEKD', type=str,
    help=('Method to evaluate: {SEKD, SEKDLarge}'))
parser.add_argument('--weights_path', type=str, default='./assets/sekd.pth',
    help='Path to pretrained weights file (default: sekd.pth).')
parser.add_argument('--max_keypoints', type=int, default=500,
    help='Maximum keypoints of detection results (default: 500).')
parser.add_argument('--nms_radius', type=int, default=4,
    help='Non Maximum Suppression (NMS) radius (default: 4).')
parser.add_argument('--conf_thresh', type=float, default=0.55,
    help='Detector confidence threshold (default: 0.55).')
parser.add_argument('--multi_scale', action='store_true',
    help='Use image pyramid to improve multi-scale ability (default: False)')
parser.add_argument('--sub_pixel_location', action='store_true',
    help='Compute the sub-pixel location of each keypoint (default: False)')
parser.add_argument('--cuda', action='store_true',
    help='Use cuda to speed up network processing speed (default: False)')

parser.add_argument('--max_length', type=int, default=9,
    help='Maximum length of point tracks (default: 9).')

parser.add_argument('--no_display', action='store_true',
    help='Do not display images no screen (default: False).')
parser.add_argument('--save_keypoints', action='store_true',
    help='Save keypoints to a directory (default: False)')
parser.add_argument('--keypoints_dir', type=str, default='keypoints_outputs/',
    help='Directory where to save keypoints (default: keypoints_outputs/).')
parser.add_argument('--save_tracks', action='store_true',
    help='Save tracking results to a directory (default: False)')
parser.add_argument('--tracks_dir', type=str, default='tracks_outputs/',
    help='Directory where to save tracking results (default: tracks_outputs/).')


def track_keypoints(args):
    print("Tracking sekd keypoints with args: {0}. \n".format(args))

    print("Init feature extractor using SEKD.")
    feature_extractor = nets.get_sekd_model(
        args.model_name, weights_path=args.weights_path,
        confidence_threshold = args.conf_thresh, nms_radius = args.nms_radius,
        max_keypoints = args.max_keypoints, cuda = args.cuda,
        multi_scale = args.multi_scale,
        sub_pixel_location = args.sub_pixel_location)

    print("Init video stream from {0}.".format(args.input))
    video_stream = Video(args.input, args.camera_id, args.img_ext)

    print("Init tracker.")
    tracker = Tracker(args.max_length)

    # Create a window to display the result.
    if not args.no_display:
        window = 'SEKD Tracker'
        cv2.namedWindow(window)
    else:
        print('Do not show the results via GUI window.')

    # Create output directory if desired.
    if args.save_keypoints:
        print('Will save keypoints to {0}.'.format(args.keypoints_dir))
        if not os.path.exists(args.keypoints_dir):
            os.makedirs(args.keypoints_dir)
    if args.save_tracks:
        print('Will save tracks to {0}.'.format(args.tracks_dir))
        if not os.path.exists(args.tracks_dir):
            os.makedirs(args.tracks_dir)

    print('Processing each frame ...')
    while True:
        # Get a new image.
        img = video_stream.next_frame()
        if img is None:
            print('All frames have been processed.')
            if not args.no_display:
                print('Press any key to quit.')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            break

        # Resize img.
        if img.shape[0] > args.max_height or img.shape[1] > args.max_width:
            resize_ratio = min(args.max_height / img.shape[0],
                args.max_width / img.shape[1])
            img = cv2.resize(img, (int(resize_ratio * img.shape[1]),
                int(resize_ratio * img.shape[0])))

        # Get points and descriptors.
        keypoints, descriptors = feature_extractor.detectAndCompute(img)

        # Save points and descriptors.
        if args.save_keypoints:
            img_name = video_stream.name_list[video_stream.i-1]
            keypoints_filepath = os.path.join(args.keypoints_dir, img_name)
            print('Save keypoints to {0}'.format(keypoints_filepath))
            np.savez(keypoints_filepath, keypoints = keypoints,
                    descriptors = descriptors)

        # Update tracks with the keypoints and descriptors.
        tracker.track(keypoints, descriptors)

        # Draw keypoint tracks on the input image.
        img_out = (img * 255.).astype('uint8')
        img_out = tracker.draw_tracks(img_out)

        # Save tracks.
        if args.save_tracks:
            img_name = video_stream.name_list[video_stream.i-1]
            img_name = str(video_stream.i-1).zfill(5) + '.png'
            tracks_filepath = os.path.join(args.tracks_dir, img_name)
            print('Save tracks to {0}'.format(tracks_filepath))
            cv2.imwrite(tracks_filepath, img_out)

            tracks_filepath = os.path.join(args.tracks_dir, img_name[:-4])
            np.savez(tracks_filepath, tracks_backward = tracker.tracks_backward[-1])

        # Display visualization image to screen.
        if not args.no_display:
            cv2.imshow(window, img_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('\'q\' has been pressed, quitting ...')
                cv2.destroyAllWindows()
                break

    print('Finshed tracking keypoints.')

if __name__ == '__main__':
    args = parser.parse_args()
    track_keypoints(args)

