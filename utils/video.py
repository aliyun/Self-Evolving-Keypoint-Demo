# Copyright (c) Alibaba Inc. All rights reserved.

import cv2
import glob
import numpy as np
import os

class Video(object):
    """ Class to read each frame from camera, video, or image sequence:"
      input_type: camera, video file path, or image sequence dir.
    """
    def __init__(self, input_type, camera_id = 0, img_ext = '.jpg'):
        self.input_type = input_type
        self.stream = None
        self.camera = False
        self.video_file = False
        self.name_list = []
        self.i = 0
        self.maxlen = int(1e7)
        if input_type == "camera":
            # If input_type is camera, then use a webcam.
            self.stream = cv2.VideoCapture(camera_id)
            self.name_list = ['{:0>6}.png'.format(x) for x in range(0, self.maxlen)]
            self.camera = True
        elif os.path.isfile(input_type):
            # Try to open a video.
            self.stream = cv2.VideoCapture(input_type)
            self.name_list = ['{:0>6}.png'.format(x) for x in range(0, self.maxlen)]
            self.video_file = True
        elif os.path.isdir(input_type):
            # Read image sequence from dir.
            for img_name in sorted(os.listdir(input_type)):
                if (img_name[-len(img_ext):] == img_ext and
                    os.path.isfile(os.path.join(input_type, img_name))):
                    self.name_list.append(img_name)
            self.maxlen = len(self.name_list)

    def next_frame(self):
        """ Read next frame and convert to gray image.
        """
        if self.i == self.maxlen:
            return None
        if self.camera:
            ret, frame = self.stream.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.video_file and self.stream.isOpened():
            ret, frame = self.stream.read()
            if ret == False:
                return None
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_path = os.path.join(self.input_type, self.name_list[self.i])
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        frame = frame.astype('float32')/255.0
        self.i = self.i + 1
        return frame

