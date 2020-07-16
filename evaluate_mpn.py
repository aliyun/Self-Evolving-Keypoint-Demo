# Copyright (c) Alibaba Inc. All rights reserved.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm_notebook as tqdm

#matplotlib inline

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

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

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
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device), 
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                    i_err_number_points[thr] += np.sum(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
                    v_err_number_points[thr] += np.sum(dist <= thr)
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_err_number_points, v_err_number_points, [seq_type, n_feats, n_matches]

def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5), 
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )

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

def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)

def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k :]
        return keypoints[ids, :], descriptors[ids, :]

cache_dir = './data/cache_mma'
if not os.path.isdir(cache_dir):
    os.makedirs(cache_dir)

errors = {}

for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print('\n' + method)
    read_function = generate_read_function(method)
    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Overall', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylabel('Mean matching accuracy', fontname='Times New Roman')
plt.ylim([0, 1])
plt.yticks([x/10 for x in range(11)], fontname='Times New Roman')
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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Illumination', fontname='Times New Roman')
plt.xlabel('Threshold [px]', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 1])
plt.yticks([x/10 for x in range(11)], fontname='Times New Roman')
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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Viewpoint', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 1])
plt.yticks([x/10 for x in range(11)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

if top_k is None:
    plt.savefig('./data/figs/hseq-mma.pdf', bbox_inches='tight', dpi=300)
else:
    plt.savefig('./data/figs/hseq-top-mma.pdf', bbox_inches='tight', dpi=300)

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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [(i_err_number_points[thr] + v_err_number_points[thr]) / ((n_i + n_v) * 5) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Overall', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylabel('Matching keypoints number', fontname='Times New Roman')
plt.ylim([0, 300])
plt.yticks([x*30 for x in range(11)], fontname='Times New Roman')
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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [i_err_number_points[thr] / (n_i * 5) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Illumination', fontname='Times New Roman')
plt.xlabel('Threshold [px]', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 300])
plt.yticks([x*30 for x in range(11)], fontname='Times New Roman')
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
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [v_err_number_points[thr] / (n_v * 5) for thr in plt_rng],
            color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Viewpoint', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylim([0, 300])
plt.yticks([x*30 for x in range(11)], fontname='Times New Roman')
plt.gca().axes.set_yticklabels([])
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)

if top_k is None:
    plt.savefig('./data/figs/hseq-mpn.pdf', bbox_inches='tight', dpi=300)
else:
    plt.savefig('./data/figs/hseq-top-mpn.pdf', bbox_inches='tight', dpi=300)

print('\n')
for method in methods:
    i_err, v_err, _, _, stats = errors[method]
    seq_type, n_feats, n_matches = stats
    accuracies = np.asarray([(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng])
    print('MMA {0}: mean: {1}, std: {2}'.format(method, np.mean(accuracies), np.std(accuracies)))

print('\n')
for method in methods:
    _, _, i_err_number_points, v_err_number_points, stats = errors[method]
    seq_type, n_feats, n_matches = stats
    matching_nums = np.asarray([(i_err_number_points[thr] + v_err_number_points[thr]) / ((n_i + n_v) * 5) for thr in plt_rng])
    print('MPN {0}: mean: {1}, std: {2}'.format(method, np.mean(matching_nums), np.std(matching_nums)))

