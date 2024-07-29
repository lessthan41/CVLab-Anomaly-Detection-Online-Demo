#!bin/bash/
python postprocess.py \
	--dataset 'visa' \
	--classes 'candle,capsules,macaroni1,macaroni2' \
	--output_path '/home/anomaly/data/segment/output' \
	--data_path '/home/anomaly/data/VisA_highshot' \
	--train \
	--reset
	# --test

python postprocess.py \
	--dataset 'mpdd' \
	--classes 'tubes' \
	--output_path '/home/anomaly/data/segment/output' \
	--data_path '/home/anomaly/data/MPDD' \
	--train \
	--reset
	# --test