# SEKD: Self-Evolving Keypoint Detection and Description

This project contains the evaluation codes and pre-trained models of SEKD.

SEKD is a general local feature algorithm using deep neural network.
To discover novel local features from image data automatically, we summarize two
natural properties of local features, i.e., repeatability and reliability.
Driven by these properties, we further design a self-evolving framework along
with training strategies to train the model effectively.

Our method mainly suffers from 3 advantages:
- Except images, we don't need any annotations or pre-processing on the data.
  This allows us to exploit arbitrary data.
- Except the summarized properties, we don't impose any restrictions on the
  learned keypoints. This allows the algorithm to discover novel features all by
  itself.
- The trained model can achieve leading performance.

For more technical details, please read our paper
["SEKD: Self-Evolving Keypoint Detection and Description"](https://arxiv.org/abs/2006.05077).

To run the evaluation experiments, please follow the guidance bellow.

## Install dependencies

We implemented SEKD using pytorch, please install it following the instructions
here: [install pytorch](https://pytorch.org/get-started/locally/).

Other python dependencies:
```
pip install numpy opencv-python opencv-contrib-python-headless \
matplotlib tqdm pillow scipy
```

## Run the experiments

live_demo_tracking.py is a live demo to detect and track SEKD features from a
camera or a video. For detail usage, run:
```
python live_deom_tracking.py -h
```

To run the experiments on HPatches dataset, please run:
```
sh run_all_evaluations.sh
```

Note that: we cached some results of D2-Net, DELF, LF-Net, and SuperPoint.
To extract the initial features, please refer to the instructions and codes in
the 3rd_party directory.

## Reference

Please cite the paper if you use this project:
```txt
@article{song_sekd,
  title = {{SEKD}: Self-Evolving Keypoint Detection and Description},
  author = {Song, Yafei and Cai, Ling and Li, Jia and Tian, Yonghong and Li, Mingyang},
  journal = {arXiv preprint arXiv: 2006.05077},
  year = {2020}
}
```

