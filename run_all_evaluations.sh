# Copyright (c) Alibaba Inc. All rights reserved.

# Download HPatches dataset.
mkdir -p data
cd ./data/
echo "\n>> Please wait, clone hpatches-benchmark\n"
git clone https://github.com/hpatches/hpatches-benchmark.git
cd ./hpatches-benchmark/
sh download.sh hpatches
sh download.sh descr sift
sh download.sh descr orb
cd ../  # at ./data

echo "\n>> Please wait, clone hpatches-dataset\n"
git clone https://github.com/hpatches/hpatches-dataset.git
cd ./hpatches-dataset/
echo "\n>> Please wait, downloading the HPatches sequences dataset ~1.3G\n"
wget -O ./hpatches-sequences-release.tar.gz http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xzf ./hpatches-sequences-release.tar.gz -C ./
rm ./hpatches-sequences-release.tar.gz
echo ">> Done!"
cd ../  # at ./data
cd ../  # at ./

# Export features on HPatches sequences images and descriptors on patches.
python ./utils/resize_imgs.py \
    --input_dir ./data/hpatches-dataset/hpatches-sequences-release \
    --output_dir ./data/hpatches-dataset/hpatches-sequences-resize
python export_results.py

# Evaluate local features: mean matching accuracy and homography estimation.
mkdir -p data/figs
python evaluate_homography.py
python evaluate_mpn.py

# Evaluate detector: keypoints repeatability.
python evaluate_repeatability.py

# Evaluate descriptor.
cd ./data/hpatches-benchmark/python/
python hpatches_eval.py --descr-name=orb --task=verification --task=matching \
	--task=retrieval --split=full
python hpatches_eval.py --descr-name=brisk --task=verification --task=matching \
	--task=retrieval --split=full
python hpatches_eval.py --descr-name=surf --task=verification --task=matching \
	--task=retrieval --split=full
python hpatches_eval.py --descr-name=sift --task=verification --task=matching \
	--task=retrieval --split=full --delimiter=';'
python hpatches_eval.py --descr-name=sekd --task=verification --task=matching \
	--task=retrieval --split=full

python hpatches_results.py --descr-name=orb --task=verification \
	--task=matching --task=retrieval --split=full
python hpatches_results.py --descr-name=brisk --task=verification \
	--task=matching --task=retrieval --split=full
python hpatches_results.py --descr-name=surf --task=verification \
	--task=matching --task=retrieval --split=full
python hpatches_results.py --descr-name=sift --task=verification \
	--task=matching --task=retrieval --split=full
python hpatches_results.py --descr-name=sekd --task=verification \
	--task=matching --task=retrieval --split=full
cd ../../../

