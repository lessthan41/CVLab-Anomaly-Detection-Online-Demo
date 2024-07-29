#!bin/bash/
export CUDA_VISIBLE_DEVICES=0
### visa
python segment.py \
	--classes 'candle' \
	--seed 888 \
	--dataset 'visa' \
	--data_path '/home/anomaly/data/VisA_highshot' \
	--output_path '/home/anomaly/data/segment/output' \
	--ckpt_filename '/home/anomaly/GroundingDINO/weights/groundingdino_swint_ogc.pth' \
	--ckpt_config_filename '/home/anomaly/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' \
	--sam_checkpoint '/home/anomaly/data/ckpt/sam_vit_h_4b8939.pth' \
	--few_shot_train \
	--gsam \
	--reset
	# --sam \
	# --full_shot_test \

# ### mpdd
# python segment.py \
# 	--classes 'tubes' \
# 	--seed 888 \
# 	--dataset 'mpdd' \
# 	--data_path '/home/anomaly/data/MPDD' \
# 	--output_path '/home/anomaly/data/segment/output' \
# 	--ckpt_filename '/home/anomaly/GroundingDINO/weights/groundingdino_swint_ogc.pth' \
# 	--ckpt_config_filename '/home/anomaly/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' \
# 	--sam_checkpoint '/home/anomaly/data/ckpt/sam_vit_h_4b8939.pth' \
# 	--few_shot_train \
# 	--sam \
# 	--gsam \
# 	--reset
# 	# --full_shot_test \


