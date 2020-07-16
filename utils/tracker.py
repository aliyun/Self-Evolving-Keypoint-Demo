# Copyright (c) Alibaba Inc. All rights reserved.

import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
import torch

class Tracker(object):
    """ Track the sparse keypoints via their descriptors.
    """
    def __init__(self, max_length, matching_method = 'nearest',
        cross_check = True, dist_thresh = 0.8, dim_descriptor = 128):
        self.max_length = max_length
        self.matching_method = matching_method
        self.cross_check = cross_check
        self.dim_descriptor = dim_descriptor
        self.dist_thresh = dist_thresh

        self.all_points = []
        self.tracks_forward = []
        self.tracks_backward = []
        for n in range(self.max_length):
            self.all_points.append(np.zeros([3, 0]))
            self.tracks_forward.append(np.zeros([2,0], dtype=np.int))
            self.tracks_backward.append(np.zeros([2,0], dtype=np.int))
        self.prev_desc = np.zeros([dim_descriptor, 0])
        self.prev_desc_index = np.zeros([2,0], dtype=np.int)

    def track(self, points, desc):
        """ Update all_points, tracks, descriptors using the newly results.
          points - 3xN array of 2D point observations.
          desc - dimxN array of dim dimensional descriptors.
        """
        # Remove oldest points and descriptors.
        self.all_points.pop(0)
        self.tracks_forward.pop(0)
        self.tracks_backward.pop(0)
        reserve_desc_id = (self.prev_desc_index[0, :] != 0)
        self.prev_desc = self.prev_desc[:, reserve_desc_id]
        self.prev_desc_index = self.prev_desc_index[:, reserve_desc_id]
        self.prev_desc_index[0, :] -= 1
        # Update the tracks.
        if points is None or desc is None:
            # No matching points, update tracks.
            self.all_points.append(np.zeros((3, 0)))
            self.tracks_forward.append(np.zeros([2,0], dtype=np.int))
            self.tracks_backward.append(np.zeros([2,0], dtype=np.int))
        else:
            correspondence_vector = self.find_correspondence(desc, self.prev_desc)

            self.all_points.append(points)
            self.tracks_forward.append(np.zeros([2,points.shape[1]], dtype=np.int) - 1)
            self.tracks_backward.append(np.zeros([2,points.shape[1]], dtype=np.int) - 1)

            reserve_desc_id = np.ones(self.prev_desc.shape[1], dtype=np.bool)
            for i in range(correspondence_vector.shape[0]):
                if correspondence_vector[i] > -1:
                    reserve_desc_id[correspondence_vector[i]] = False
                    i_frame_prev = self.prev_desc_index[0, correspondence_vector[i]]
                    i_point_prev = self.prev_desc_index[1, correspondence_vector[i]]
                    self.tracks_backward[-1][0, i] = (
                         i_frame_prev - len(self.all_points) + 1)
                    self.tracks_backward[-1][1, i] = i_point_prev
                    self.tracks_forward[i_frame_prev][0,i_point_prev] = (
                         len(self.all_points) - i_frame_prev - 1)
                    self.tracks_forward[i_frame_prev][1,i_point_prev] = i
            self.prev_desc = self.prev_desc[:, reserve_desc_id]
            self.prev_desc_index = self.prev_desc_index[:, reserve_desc_id]
            self.prev_desc = np.concatenate((self.prev_desc, desc), axis=1)
            for i in range(points.shape[1]):
                self.prev_desc_index = np.concatenate(
                    (self.prev_desc_index, np.array(
                    [[len(self.all_points)-1], [i]], dtype=np.int)), axis=1)
        return

    def find_correspondence(self, desc_a, desc_b):
        correspondence_vector = np.zeros(desc_a.shape[1], dtype=np.int) - 1
        if desc_a.shape[1] == 0 or desc_b.shape[1] == 0:
            return correspondence_vector
        #distance_matrix = cdist(desc_a.T, desc_b.T)
        distance_matrix = torch.cdist(
            torch.tensor(desc_a.astype(np.float32).T).unsqueeze(0).cuda(),
            torch.tensor(desc_b.astype(np.float32).T).unsqueeze(0).cuda()
            ).squeeze(0).cpu().numpy()
        if self.matching_method == 'nearest':
            min_index_a2b = np.argmin(distance_matrix, axis=1)
            correspondence_vector[:] = min_index_a2b[:]
            if self.cross_check:
                min_index_b2a = np.argmin(distance_matrix, axis=0)
                all_index_a = np.array([i for i in range(desc_a.shape[1])])
                cross_check = (min_index_b2a[min_index_a2b[all_index_a]] == all_index_a)
                correspondence_vector[:] = -1
                correspondence_vector[cross_check] = min_index_a2b[cross_check]
            if self.dist_thresh > 0:
                min_distance_a2b = np.min(distance_matrix, axis=1)
                min_distance_discard = (min_distance_a2b > self.dist_thresh)
                correspondence_vector[min_distance_discard] = -1
        else:
            print("[Error] now only support nearest matching.")
        return correspondence_vector

    def draw_tracks(self, img_rgb):
        assert(len(img_rgb.shape) == 3)
        assert(img_rgb.shape[2] == 3)
        for i_track in range(self.tracks_backward[-1].shape[1]):
            if self.tracks_backward[-1][1, i_track] > -1:
                point = (self.all_points[-1][0, i_track],
                    self.all_points[-1][1, i_track])
                cv2.circle(img_rgb, point, 1, [0., 0., 255.], -1, lineType=16)
                i_frame = len(self.all_points) - 1
                i_point = i_track
                track_length = 1
                while self.tracks_backward[i_frame][1, i_point] > -1:
                    track_length += 1
                    i_frame_prev = i_frame + self.tracks_backward[i_frame][0, i_point]
                    i_point_prev = self.tracks_backward[i_frame][1, i_point]
                    if i_frame_prev < 0:
                        break
                    point_prev = (self.all_points[i_frame_prev][0, i_point_prev],
                        self.all_points[i_frame_prev][1, i_point_prev])
                    cv2.line(img_rgb, point, point_prev, [0., 64., 255.], thickness=1, lineType=16)
                    point = point_prev
                    i_frame = i_frame_prev
                    i_point = i_point_prev
        return img_rgb

