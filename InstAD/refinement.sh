#!bin/bash/
export CUDA_VISIBLE_DEVICES=0
# python instance_segment.py \
# 	--dataset 'mpdd' \
# 	--data_path '/home/anomaly/data/MPDD' \
# 	--working_directory '/home/anomaly/research/i-patchcore/segment/' \
# 	--output_path '/home/anomaly/data/segment/output' \
# 	--classes 'tubes' \
# 	--sam_checkpoint '/home/anomaly/data/ckpt/sam_vit_b_01ec64.pth' \
# 	--cluster_batch_size 20 \
# 	--train \
# 	--demo
	# --reset \
	# --refine_segment \
	# --test
	# --sam_checkpoint '/home/anomaly/data/ckpt/sam_vit_h_4b8939.pth' \

python instance_segment.py \
	--dataset 'visa' \
	--data_path '/home/anomaly/data/VisA_highshot' \
	--working_directory '/home/anomaly/research/i-patchcore/segment' \
	--output_path '/home/anomaly/data/segment/output' \
	--classes 'capsules' \
	--sam_checkpoint '/home/anomaly/data/ckpt/sam_vit_b_01ec64.pth' \
	--cluster_batch_size 20 \
	--feature_alignment \
	--refine_segment \
	--test \
	--reset \
	--demo
	# --test
