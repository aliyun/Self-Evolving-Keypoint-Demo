# Copyright (c) Alibaba Inc. All rights reserved.

import cv2
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

def benchmark_features(read_feats, method=None):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        img_ref = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'), cv2.IMREAD_GRAYSCALE)
        height_ref, width_ref = img_ref.shape
        corner_a_h = np.array([[0, 0, 1], [width_ref-1, 0, 1],
                             [0, height_ref-1, 1], [width_ref-1, height_ref-1, 1]])
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])
        for im_idx in range(2, 7):
            homography_gt = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])
            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )
            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            pos_a = keypoints_a[matches[:, 0], :2]
            pos_b = keypoints_b[matches[:, 1], :2]
            homography_es, mask = cv2.findHomography(pos_a, pos_b, cv2.RANSAC)
            if homography_es is None:
                continue

            corner_b_h_proj_gt = np.transpose(np.dot(homography_gt, np.transpose(corner_a_h)))
            corner_b_proj_gt = corner_b_h_proj_gt[:, :2] / corner_b_h_proj_gt[:, 2:]
            corner_b_h_proj_es = np.transpose(np.dot(homography_es, np.transpose(corner_a_h)))
            corner_b_proj_es = corner_b_h_proj_es[:, :2] / corner_b_h_proj_es[:, 2:]
            dist = np.mean(np.sqrt(np.sum((corner_b_proj_gt - corner_b_proj_es) ** 2, axis=1)))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += (dist <= thr)
                else:
                    v_err[thr] += (dist <= thr)
            # Visualize results.
            #print(os.path.join(dataset_path, seq_name, str(im_idx)+'.ppm'))
            img_target = cv2.imread(os.path.join(dataset_path, seq_name, str(im_idx)+'.ppm'), cv2.IMREAD_GRAYSCALE)
            height_target, width_target = img_target.shape
            img_match = np.zeros([max(height_ref, height_target), width_ref+width_target, 3])
            for i_channel in range(3):
                img_match[0:height_ref,0:width_ref,i_channel] = img_ref
                img_match[0:height_target,width_ref::,i_channel] = img_target
            # Draw keypoints.
            for i_keypoints in range(keypoints_a.shape[0]):
                cv2.circle(img_match, (int(keypoints_a[i_keypoints, 0]), int(keypoints_a[i_keypoints, 1])), radius=0, color=(0,0,255), thickness=2)
            for i_keypoints in range(keypoints_b.shape[0]):
                cv2.circle(img_match, (int(keypoints_b[i_keypoints, 0])+width_ref, int(keypoints_b[i_keypoints, 1])), radius=0, color=(0,0,255), thickness=2)
            # Draw keypoints matches.
            for i_matches in range(matches.shape[0]):
                cv2.line(img_match,
                        (int(keypoints_a[matches[i_matches, 0], 0]), int(keypoints_a[matches[i_matches, 0], 1])),
                        (int(keypoints_b[matches[i_matches, 1], 0])+width_ref, int(keypoints_b[matches[i_matches, 1], 1])),
                        color=(0,255,0), thickness=1)
            if not os.path.isdir(os.path.join('./data/hpatches_stitched', seq_name)):
                os.makedirs(os.path.join('./data/hpatches_stitched', seq_name))
            cv2.imwrite(os.path.join('./data/hpatches_stitched', seq_name, "matched_1_{0}_{1}.jpg".format(im_idx, method)), img_match)
            # Draw stitch result.
            img_stitched = cv2.warpPerspective(img_ref, homography_es, (width_ref+width_target, max(height_ref, height_target)))
            img_stitched[0:height_target, 0:width_target] = img_stitched[0:height_target, 0:width_target] / 2 + img_target / 2
            img_stitched = img_stitched[0:height_target, 0:width_target]
            cv2.imwrite(os.path.join('./data/hpatches_stitched', seq_name, "stitched_1_{0}_{1}.jpg".format(im_idx, method)), img_stitched)
#            break
#        break
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]

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

cache_dir = './data/cache_homo'
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
        print('Computing errors...')
        errors[method] = benchmark_features(read_function, method)
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
    i_err, v_err, stats = errors[method]
    seq_type, n_feats, n_matches = stats
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
            for thr in plt_rng], color=color, ls=ls, marker=marker, linewidth=2, label=name)
plt.title('Overall', fontname='Times New Roman')
plt.xlim(plt_lim)
plt.xticks(plt_rng, fontname='Times New Roman')
plt.ylabel('Homography accuracy', fontname='Times New Roman')
plt.ylim([0, 1])
plt.yticks([x/10 for x in range(11)], fontname='Times New Roman')
plt.grid(linestyle=':')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop=font1)

plt.subplot(1, 3, 2)
for method in methods:
    name = methods[method]['name']
    color = methods[method]['color']
    ls = methods[method]['linestyle']
    marker = methods[method]['marker']
    i_err, v_err, stats = errors[method]
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
    i_err, v_err, stats = errors[method]
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
    plt.savefig('./data/figs/hseq-homo.pdf', bbox_inches='tight', dpi=300)
else:
    plt.savefig('./data/figs/hseq-top-homo.pdf', bbox_inches='tight', dpi=300)

print('\n')
for method in methods:
    i_err, v_err, stats = errors[method]
    seq_type, n_feats, n_matches = stats
    accuracy = np.asarray([(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng])
    accuracy_i = np.asarray([(i_err[thr]) / ((n_i) * 5) for thr in plt_rng])
    accuracy_v = np.asarray([(v_err[thr]) / ((n_v) * 5) for thr in plt_rng])
    print('{0}: {1}, {2}, {3}'.format(method, np.mean(accuracy), np.mean(accuracy_i), np.mean(accuracy_v)))

