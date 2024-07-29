import os
import random
import shutil
import tqdm
import warnings
import argparse
from .segment.gsam import groundingdino, load_model_hf
from .segment.sam import segment_anything
from .segment.utils import reset_folder
from .config import amg_kwargs
from segment_anything import sam_model_registry, SamPredictor
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

def load_sam(sam_checkpoint, model_type, device='cuda'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.eval().to(device=device)
    print("successfully load SAM============================================")
    return sam

def load_groundingdino(sam, ckpt_filename, ckpt_config_filename, device='cuda'):
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    groundingdino_model = load_model_hf(ckpt_filename, ckpt_config_filename, device)
    sam_predictor = SamPredictor(sam)
    groundingdino_model.eval()
    print("successfully load groundingdino==================================")
    return groundingdino_model, sam_predictor

def main(model_type, device, amg_kwargs, args):
    reset = False
    data_path = f"{args.data_path}"
    img_output = f"{args.output_path}/{args.dataset}/image/"
    sam_mask_output = f"{args.output_path}/{args.dataset}/sam_mask/"
    gsam_mask_output = f"{args.output_path}/{args.dataset}/gsam_mask/"

    models = []
    if args.sam:
        models.append("sam")
    if args.gsam:
        models.append("gsam")
    
    for model in models:
        print(f"Model: {model}")
        ### loading model
        print(f"Loading {model} model...")
        if model == "sam":
            sam = load_sam(args.sam_checkpoint, model_type, device)
        elif model == "gsam":
            sam = load_sam(args.sam_checkpoint, model_type, device)
            groundingdino_model, sam_predictor = load_groundingdino(sam, args.ckpt_filename, args.ckpt_config_filename, device)
        
        for class_name in os.listdir(data_path):
            if class_name not in args.classes.split(","):
                continue

            box_thres = 0.3
            ### training image
            if args.one_shot_train or args.few_shot_train or args.full_shot_train:
                input_dir = os.path.join(data_path,class_name,"train","good")
                output_dir = os.path.join(img_output,class_name,"train","good")
                sam_output_dir = os.path.join(sam_mask_output,class_name,"train","good")
                gsam_output_dir = os.path.join(gsam_mask_output,class_name,"train","good")
                if not reset:
                    reset_folder(output_dir)
                    reset = True
                else:
                    os.makedirs(output_dir, exist_ok=True)
                if args.one_shot_train:
                    ### training images (oneshot)
                    if len(os.listdir(output_dir)) == 0:
                        img_list = os.listdir(input_dir)
                        img_path_list = random.choice(img_list)
                    else:
                        img_path_list = os.listdir(output_dir)
                elif args.few_shot_train:    
                    ### training images (fewshot)
                    if len(os.listdir(output_dir)) == 0:
                        img_list = os.listdir(input_dir)
                        img_path_list = random.choices(img_list,k=8)
                    else:
                        img_path_list = os.listdir(output_dir)
                elif args.full_shot_train:
                    ### training images (fullshot)
                    img_path_list = os.listdir(input_dir)

                if not os.path.exists(sam_output_dir):
                    reset_folder(sam_output_dir)
                if not os.path.exists(gsam_output_dir):
                    reset_folder(gsam_output_dir)
                if args.reset:
                    if model == "sam":
                        reset_folder(sam_output_dir)
                    elif model == "gsam":
                        reset_folder(gsam_output_dir)

                ### resume
                if model == "sam":
                    curr_img_list = os.listdir(sam_output_dir)
                elif model == "gsam":
                    curr_img_list = os.listdir(gsam_output_dir)
                for path in tqdm.tqdm(img_path_list):
                    if path in curr_img_list:
                        continue
                    img_path = os.path.join(data_path,class_name,"train","good",path)
                    shutil.copyfile(img_path,os.path.join(img_output,class_name,"train","good",os.path.basename(img_path)))
                    if model == "sam":
                        segment_anything(sam,img_path,sam_output_dir,model_type,device,amg_kwargs)
                    elif model == "gsam":
                        groundingdino(input_path=img_path, output_path=gsam_output_dir, class_name=class_name, box_threshold=box_thres, groundingdino_model=groundingdino_model, sam_predictor=sam_predictor, is_invert=False)

            ### testing image
            if args.full_shot_test:
                ### testing images
                for defect in os.listdir(os.path.join(data_path,class_name,"test")):
                    input_dir = os.path.join(data_path,class_name,"test",defect)
                    output_dir = os.path.join(img_output,class_name,"test",defect)
                    sam_output_dir = os.path.join(sam_mask_output,class_name,"test",defect)
                    gsam_output_dir = os.path.join(gsam_mask_output,class_name,"test",defect)
                    img_list = os.listdir(input_dir)
                    img_path = [os.path.join(input_dir,x) for x in img_list]

                    if not os.path.exists(sam_output_dir):
                        reset_folder(sam_output_dir)
                    if not os.path.exists(gsam_output_dir):
                        reset_folder(gsam_output_dir)
                    if args.reset:
                        if model == "sam":
                            reset_folder(sam_output_dir)
                        elif model == "gsam":
                            reset_folder(gsam_output_dir)
                    
                    ### resume
                    if model == "sam":
                        curr_img_list = os.listdir(sam_output_dir)
                    elif model == "gsam":
                        curr_img_list = os.listdir(gsam_output_dir)
                    for path in tqdm.tqdm(img_path):
                        if os.path.basename(path) in curr_img_list:
                            continue
                        if model == "sam":
                            segment_anything(sam,path,sam_output_dir,model_type,device,amg_kwargs)
                        elif model == "gsam":
                            groundingdino(input_path=path, output_path=gsam_output_dir, class_name=class_name, box_threshold=box_thres, groundingdino_model=groundingdino_model, sam_predictor=sam_predictor, is_invert=False)

        if model == "sam":
            del sam
        elif model == "gsam":
            del groundingdino_model
            del sam_predictor

    print("Done!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument('-c', '--classes', type=str, help='classes for segmentation', default="capsules")
    parser.add_argument('-s', '--seed', type=int, help='seed number', default=777)
    parser.add_argument('-d', '--dataset', type=str, help='dataset', default="visa")
    parser.add_argument('--data_path', type=str, help='data path', default="/mnt/sda/anomaly/VisA_highshot/")
    parser.add_argument('--output_path', type=str, help='output path', default="/mnt/sda/anomaly/segment/output")
    parser.add_argument('--sam_checkpoint', type=str, help='sam checkpoint', default="/mnt/sda/anomaly/sam_ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--ckpt_filename", type=str, help="checkpoint filename", default="/home/tokichan/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--ckpt_config_filename", type=str, help="checkpoint config filename", default="/home/tokichan/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--one_shot_train', action='store_true', help='one shot train', default=False)
    parser.add_argument('--few_shot_train', action='store_true', help='few shot train', default=False)
    parser.add_argument('--full_shot_train', action='store_true', help='full shot train', default=False)
    parser.add_argument('--full_shot_test', action='store_true', help='full shot test', default=False)
    parser.add_argument('--sam', action='store_true', help='run sam', default=False)
    parser.add_argument('--gsam', action='store_true', help='run gsam', default=False)
    parser.add_argument('--reset', action='store_true', help='rerun everything (False for resume)', default=False)
    args = parser.parse_args()
    
    ### Seed
    random.seed(args.seed)
    model_type = "vit_h" if "vit_h" in args.sam_checkpoint else "vit_b"
    device = "cuda"

    main(model_type, device, amg_kwargs, args)
