# Copyright (c) Alibaba Inc. All rights reserved.

import argparse
import cv2
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import shutil
import sys
from tqdm import tqdm_notebook as tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Add new methods here.
methods = {
    #'d2-net':{'name':'D2-Net','color':'purple', 'linestyle':':', 'marker':'+'},
    'orb':   {'name':'ORB',   'color':'black',  'linestyle':'-.','marker':'d'},
    #'delf':  {'name':'DELF',  'color':'pink',   'linestyle':'--','marker':'x'},
    'akaze': {'name':'AKAZE', 'color':'magenta','linestyle':'-', 'marker':'*'},
    'brisk': {'name':'BRISK', 'color':'cyan',   'linestyle':':', 'marker':'p'},
    'surf':  {'name':'SURF',  'color':'red',    'linestyle':'-.','marker':'s'},
    'kaze':  {'name':'KAZE',  'color':'gray',   'linestyle':'--','marker':'>'},
    #'lfnet': {'name':'LF-Net','color':'brown',  'linestyle':'-', 'marker':'<'},
    'sift':  {'name':'SIFT',  'color':'orange', 'linestyle':':', 'marker':'^'},
    #'superpoint':{'name':'SuperPoint','color':'blue','linestyle':'-.','marker':'v'},
    'sekd':  {'name':'SEKD',  'color':'green',  'linestyle':'-', 'marker':'o'},
    }

font1={'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10}

# Change here if you want to use top K or all features.
top_k = 500
#top_k = None

n_i = 57
n_v = 59

dataset_path = './data/hpatches-dataset/hpatches-sequences-resize'
features_path = './data/hpatches-dataset/features'

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def benchmark_features(read_feats):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}
    i_err_number_points = {thr: 0 for thr in rng}
    v_err_number_points = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, _ = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, _ = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            homography_inverse = np.linalg.inv(homography)

            pos_a = keypoints_a[:, : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([keypoints_a.shape[0], 1])], axis=1)
            pos_a_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_a_proj = pos_a_proj_h[:, : 2] / pos_a_proj_h[:, 2 :]

            pos_b = keypoints_b[:, : 2]
            pos_b_h = np.concatenate([pos_b, np.ones([keypoints_b.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography_inverse, np.transpose(pos_b_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

            dist_matrix_a = cdist(pos_a, pos_b_proj)
            min_dist_a = np.min(dist_matrix_a, axis=1)
            min_dist_a_index = np.argmin(dist_matrix_a, axis=1)
            dist_matrix_b = cdist(pos_b, pos_a_proj)
            min_dist_b = np.min(dist_matrix_b, axis=1)
            min_dist_b_index = np.argmin(dist_matrix_b, axis=1)

            # Cross check.
            all_index_a = np.array([i for i in range(keypoints_a.shape[0])])
            all_index_b = np.array([i for i in range(keypoints_b.shape[0])])
            cross_check_a = (min_dist_b_index[min_dist_a_index[all_index_a]] == all_index_a)
            cross_check_b = (min_dist_a_index[min_dist_b_index[all_index_b]] == all_index_b)

            seq_type.append(seq_name[0])

            for thr in rng:
                num_rep = (np.sum((min_dist_a < thr) & cross_check_a)
                        + np.sum((min_dist_b < thr) & cross_check_b))
                if seq_name[0] == 'i':
                    i_err_number_points[thr] += num_rep
                    i_err[thr] += (num_rep / (keypoints_a.shape[0] + keypoints_b.shape[0]))
                else:
                    v_err_number_points[thr] += num_rep
                    v_err[thr] += (num_rep / (keypoints_a.shape[0] + keypoints_b.shape[0]))
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)

    return i_err, v_err, i_err_number_points, v_err_number_points, [seq_type, n_feats]

def summary(stats):
    seq_type, n_feats = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))

def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(features_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k :]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
    return read_function

cache_dir = './data/cache_rep'
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

errors = {}

for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print('\n' + method)
    read_function = generate_read_function(method, extension='ppm')
    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
        print('Computing errors...')
        errors[method] = benchmark_features(read_function)
        np.save(output_file, errors[method])
    summary(errors[method][-1])

plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(15, 4.5))

plt.subplot(1, 3, 1)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    i_err, v_err, _, _, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Overall', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylabel('Detector repeatability', fontname='Times New Roman')
plt.ylim([0, 0.8])
plt.yticks([x/10 for x in range(9)], fontname='Times New Roman')
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop=font1, loc='lower right')

plt.subplot(1, 3, 2)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    i_err, v_err, _, _, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Illumination', fontname='Times New Roman')
plt.xlabel('Threshold [px]', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 0.8])
plt.yticks([x/10 for x in range(9)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    i_err, v_err, _, _, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Viewpoint', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 0.8])
plt.yticks([x/10 for x in range(9)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

if top_k is None:
    plt.savefig('./data/figs/hseq-mrep.pdf', bbox_inches='tight', dpi=300)
else:
    plt.savefig('./data/figs/hseq-top-mrep.pdf', bbox_inches='tight', dpi=300)

#-------------------------------------------------------------------------------
plt.cla()

plt.figure(figsize=(15, 4.5))

plt.subplot(1, 3, 1)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    _, _, i_err_number_points, v_err_number_points, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [(i_err_number_points[thr] + v_err_number_points[thr]) / 2. / ((n_i + n_v) * 5.) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Overall', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylabel('Repeated keypoints number', fontname='Times New Roman')
plt.ylim([0, 400])
plt.yticks([x*40 for x in range(11)], fontname='Times New Roman')
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop=font1, loc='lower right')

plt.subplot(1, 3, 2)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    _, _, i_err_number_points, v_err_number_points, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [i_err_number_points[thr] / 2. / (n_i * 5) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Illumination', fontname='Times New Roman')
plt.xlabel('Threshold [px]', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 400])
plt.yticks([x*40 for x in range(11)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    _, _, i_err_number_points, v_err_number_points, stats = errors[method]
    seq_type, n_feats = stats
    plt.plot(plt_rng, [v_err_number_points[thr] / 2. / (n_v * 5) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Viewpoint', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 400])
plt.yticks([x*40 for x in range(11)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

if top_k is None:
    plt.savefig('./data/figs/hseq-mrepn.pdf', bbox_inches='tight', dpi=300)
else:
    plt.savefig('./data/figs/hseq-top-mrepn.pdf', bbox_inches='tight', dpi=300)

print('\n')
for method in methods:
    i_err, v_err, _, _, stats = errors[method]
    seq_type, n_feats = stats
    accuracies = np.asarray([(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng])
    print('MMA {0}: mean: {1}, std: {2}'.format(method, np.mean(accuracies), np.std(accuracies)))

print('\n')
for method in methods:
    _, _, i_err_number_points, v_err_number_points, stats = errors[method]
    seq_type, n_feats = stats
    matching_nums = np.asarray([(i_err_number_points[thr] + v_err_number_points[thr]) / 2. / ((n_i + n_v) * 5) for thr in plt_rng])
    print('MPN {0}: mean: {1}, std: {2}'.format(method, np.mean(matching_nums), np.std(matching_nums)))

