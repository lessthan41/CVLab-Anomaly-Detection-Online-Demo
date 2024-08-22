import os
from multiprocessing import Pool

dataset_classes = {
    "visa": ["candle", "capsules", "macaroni1", "macaroni2"],
    "mpdd": ["tubes"],
}

### config
visa_path = "/home/anomaly/data/VisA_highshot"
segment_output_path = "/home/anomaly/data/segment/output"
dino_ckpt_filename = "/home/anomaly/GroundingDINO/weights/groundingdino_swint_ogc.pth"
dino_ckpt_config_filename = "/home/anomaly/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
sam_ckpt = "/home/anomaly/segment-anything/ckpt/sam_vit_h_4b8939.pth"
###

if __name__ == '__main__':
    pool = Pool(processes=1)
    datasets = ['visa']
    classes = ["candle"]
    for dataset in datasets:
        classes = dataset_classes[dataset] if classes==[] else classes
        for cls in classes:
            sh_method = f"python ./segment/segment.py " \
            f"--classes '{cls}' " \
            f"--seed 888 " \
            f"--dataset '{dataset}' " \
            f"--data_path '{visa_path}' " \
            f"--output_path '{segment_output_path}' " \
            f"--ckpt_filename '{dino_ckpt_filename}' " \
            f"--ckpt_config_filename '{dino_ckpt_config_filename}' " \
            f"--sam_checkpoint '{sam_ckpt}' " \
            f"--few_shot_train " \
            # f"--full_shot_test " \
            f"--sam " \
            f"--gsam " \
            f"--reset "
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

            sh_method = f"python ./segment/postprocess.py " \
            f"--classes '{cls}' " \
            f"--dataset '{dataset}' " \
            f"--data_path '{visa_path}' " \
            f"--output_path '{segment_output_path}' " \
            f"--train " \
            # f"--test " \
            f"--reset "
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

