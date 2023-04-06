
python -W ignore sekd_train_em.py \
	--dataset_config config/dataset_config_coco_val.yaml \
	--num_refs 12 --num_workers 4 \
	--detector_loss focal_loss --confidence_threshold_detector 0.46 \
	--confidence_threshold_reliability 0.85 \
	--iterations_em 5 --epoches_detector 10 --epoches_descriptor 10 \
	--nms_radius 3 --height 512 --width 512 \
	--model_name SEKD \
	--batch_size 2 --lr 1e-4 --weight_decay 1e-8 --sub_set 5
