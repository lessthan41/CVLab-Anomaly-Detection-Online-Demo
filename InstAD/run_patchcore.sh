#!/bin/bash
# instpath=/home/tokichan/data/segment/output/visa/instance
instpath=/home/tokichan/data/segment/output/mpdd/instance
# datapath=/home/tokichan/data/VisA_highshot
datapath=/home/tokichan/data/MPDD
outpath=/home/tokichan/data/visualize/patchcore
# datasets=('candle' 'capsules' 'macaroni1' 'macaroni2')
datasets=('tubes')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

# visa zeroshot
# python run_patchcore.py --gpu 0 --seed 0 --zero_shot --image_output_path $outpath --data_path $datapath --log_group zeroshot --log_project VisA_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 1 --patchsize 3 --dist_metric L2 my_sampler identity dataset --resize 256 --batch_size 16 "${dataset_flags[@]}" visa $instpath

# visa zeroshot (output image)
# python run_patchcore.py --gpu 0 --seed 0 --zero_shot --output_image --image_output_path $outpath --data_path $datapath --log_group zeroshot --log_project VisA_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 1 --patchsize 3 --dist_metric L2 my_sampler identity dataset --resize 256 --batch_size 16 "${dataset_flags[@]}" visa $instpath

# visa fewshot
python run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --image_output_path $outpath --data_path $datapath --log_group fewshot --log_project VisA_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 1 --patchsize 3 --dist_metric L2 my_sampler identity dataset --resize 256 --k_shot 4 "${dataset_flags[@]}" visa $instpath

# visa fewshot (output image)
# python run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --output_image --image_output_path $outpath --data_path $datapath --log_group fewshot --log_project VisA_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 1 --patchsize 3 --dist_metric L2 my_sampler identity dataset --resize 256 --k_shot 4 "${dataset_flags[@]}" visa $instpath

# visa fewshot augment
# python run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --image_output_path $outpath --data_path $datapath --log_group fewshot --log_project VisA_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 512 --anomaly_scorer_num_nn 1 --patchsize 3 --dist_metric L2 my_sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --k_shot 4 --augment "${dataset_flags[@]}" visa $instpath
