import cv2
import os
import numpy as np
import tqdm
import argparse
from .segment.utils import *

def postprocess_steps(src="",dst="",class_name="", mask=None, save_masks=True):
    if save_masks:
        img = cv2.imread(src)
    else:
        img = mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = check_and_invert_mask(img, background=True)
    img = np.stack([img,img,img],axis=2)
    if save_masks:
        cv2.imwrite(dst, img)
    return img

def postorocess(input_folder, output_folder, args):
    input = f"{args.output_path}/{args.dataset}/{input_folder}"
    output = f"{args.output_path}/{args.dataset}/{output_folder}"

    print(f"gsam mask postprocessing...")
    for class_name in os.listdir(input):
        if class_name not in args.classes.split(","):
            continue
        print(class_name)
        if args.train:
            input_dir = os.path.join(input,class_name,"train","good")
            output_dir = os.path.join(output,class_name,"train","good")
            if args.reset:
                reset_folder(output_dir)
            ### resume
            os.makedirs(output_dir, exist_ok=True)
            curr_img_list = os.listdir(output_dir)
            for img_name in os.listdir(input_dir):
                if img_name in curr_img_list:
                    continue
                postprocess_steps(os.path.join(input_dir,img_name), 
                            os.path.join(output_dir,img_name), 
                            class_name)
        if args.test:
            input_dir = os.path.join(input,class_name,"test")
            output_dir = os.path.join(output,class_name,"test")
            for defect in tqdm.tqdm(os.listdir(input_dir)):
                if args.reset:
                    reset_folder(os.path.join(output_dir,defect))
                ### resume
                os.makedirs(os.path.join(output_dir,defect), exist_ok=True)
                curr_img_list = os.listdir(os.path.join(output_dir,defect))
                for img_name in os.listdir(os.path.join(input_dir,defect)):
                    if img_name in curr_img_list:
                        continue
                    postprocess_steps(os.path.join(input_dir,defect,img_name), 
                                os.path.join(output_dir,defect,img_name),
                                class_name)
    print(f"gsam mask postprocess done!")
    return

def intersect_steps(gsam_src="", sam_src="", dst="", threshold=0.9, save_masks=True):
    if save_masks:
        gsam_mask = cv2.imread(gsam_src)[:,:,0]
        sam_masks = load_pickle(sam_src)
    else:
        gsam_mask = gsam_src[:,:,0]
        sam_masks = sam_src
    gsam_mask = dilate(gsam_mask, ksize=5)
    final_mask = np.zeros_like(gsam_mask)
    for mask in sam_masks:
        ### calculate overlap ratio
        ratio = overlap_ratio(mask, gsam_mask)
        ### threshold
        if ratio >= threshold:
            mask = mask.astype(np.uint8) * 255
            final_mask = np.logical_or(final_mask, mask)

    final_mask = final_mask.astype(np.uint8) * 255
    final_mask = check_and_invert_mask(final_mask, background=False)
    final_mask = fill_border(final_mask)
    final_mask = closing(final_mask, ksize=5)
    final_mask = canny(final_mask, low=50, high=150)
    final_mask = fill_small_instances(final_mask)
    final_mask = remove_noise(final_mask)
    final_mask = np.stack([final_mask,final_mask,final_mask], axis=2)
    if save_masks:
        cv2.imwrite(dst, final_mask)
    return final_mask

def intersect(gsam_input_folder, sam_input_folder, output_folder, args, background=False):
    gsam_input = f"{args.output_path}/{args.dataset}/{gsam_input_folder}"
    sam_input = f"{args.output_path}/{args.dataset}/{sam_input_folder}"
    output = f"{args.output_path}/{args.dataset}/{output_folder}"

    print("gsam/sam intersecting...")
    for class_name in os.listdir(gsam_input):
        if class_name not in args.classes.split(","):
            continue
        print(class_name)
        if args.train:
            gsam_input_dir = os.path.join(gsam_input,class_name,"train","good")
            sam_input_dir = os.path.join(sam_input,class_name,"train","good")
            output_dir = os.path.join(output,class_name,"train","good")
            if args.reset:
                reset_folder(output_dir)
            ### resume
            os.makedirs(output_dir, exist_ok=True)
            curr_img_list = os.listdir(output_dir)
            # curr_img_list = []
            for img_name in os.listdir(gsam_input_dir):
                if img_name in curr_img_list:
                    continue
                if not os.path.exists(sam_input_dir):
                    continue
                if img_name not in os.listdir(sam_input_dir):
                    continue
                intersect_steps(
                    gsam_src=os.path.join(gsam_input_dir,img_name),
                    sam_src=os.path.join(sam_input_dir,img_name[:-4]+".pkl"), 
                    dst=os.path.join(output_dir,img_name), 
                    threshold=0.95
                )
        if args.test:
            gsam_input_dir = os.path.join(gsam_input,class_name,"test")
            sam_input_dir = os.path.join(sam_input,class_name,"test")
            output_dir = os.path.join(output,class_name,"test")
            for defect in os.listdir(gsam_input_dir):
                if args.reset:
                    reset_folder(os.path.join(output_dir,defect))
                ### resume
                os.makedirs(os.path.join(output_dir,defect), exist_ok=True)
                curr_img_list = os.listdir(os.path.join(output_dir,defect))
                # curr_img_list = []
                for img_name in tqdm.tqdm(os.listdir(os.path.join(gsam_input_dir,defect))):
                    if img_name in curr_img_list:
                        continue
                    if not os.path.exists(os.path.join(sam_input_dir,defect)):
                        continue
                    if img_name not in os.listdir(os.path.join(sam_input_dir,defect)):
                        continue
                    intersect_steps(
                        gsam_src=os.path.join(gsam_input_dir,defect,img_name),
                        sam_src=os.path.join(sam_input_dir,defect,img_name[:-4]+".pkl"), 
                        dst=os.path.join(output,class_name,"test",defect,img_name),
                        threshold=0.95
                    )
    print("gsam/sam intersect done!")
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument('-d', '--dataset', type=str, help='dataset', default='mpdd')
    parser.add_argument('-c', '--classes', type=str, help='classes for segmentation', default="capsules")
    parser.add_argument('--output_path', type=str, help='output path', default='/mnt/sda/anomaly/segment/output')
    parser.add_argument('--data_path', type=str, help='data path', default='/mnt/sda/anomaly/MPDD')
    parser.add_argument('--train', action='store_true', help='training set postprocess', default=False)
    parser.add_argument('--test', action='store_true', help='testing set postprocess', default=False)
    parser.add_argument('--reset', action='store_true', help='rerun everything (False for resume)', default=False)
    args = parser.parse_args()
    
    ### gsam postprocess
    input_folder = "gsam_mask"
    output_folder = "gsam_mask_postprocess"
    postorocess(input_folder, output_folder, args)

    ### gsam sam background intersect
    gsam_input_folder = "gsam_mask_postprocess"
    sam_input_folder = "sam_mask"
    output_folder = "gsam_sam_intersect"
    intersect(gsam_input_folder, sam_input_folder, output_folder, args)