import os
import cv2
import numpy as np
from gsam import gsam, load_model_hf
from segment_anything import sam_model_registry, SamPredictor

CKPT_REPO_ID = "ShilongLiu/GroundingDINO"
CKPT_FILENAME = "/home/tokichan/GroundingDINO/weights/groundingdino_swint_ogc.pth"
CKPT_CONFIG_FILENAME = "/home/tokichan/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
device = "cuda"
model_type = "vit_b"
sam_checkpoint = "/home/tokichan/segment-anything/ckpt/sam_vit_b_01ec64.pth"
# sam_checkpoint = "/home/tokichan/segment-anything/ckpt/sam_vit_h_4b8939.pth"
box_thres=0.1
img_paths = [
    "/home/tokichan/research/i-patchcore/src/segment/output/trash1.jpg",
    "/home/tokichan/research/i-patchcore/src/segment/output/trash2.jpg",
    "/home/tokichan/research/i-patchcore/src/segment/output/trash3.jpg",
    "/home/tokichan/research/i-patchcore/src/segment/output/trash4.jpg",
]
output_dir = "/home/tokichan/research/i-patchcore/src/segment/"


def check_and_invert_mask(mask, background=False):
    """ check the mask is background or not, and invert if necessary """
    ret_mask = mask
    total_pixels = mask.size
    white_pixels = np.sum(mask == 255)
    white_proportion = white_pixels / total_pixels
    if not background:
        if white_proportion > 0.5:
            ret_mask = cv2.bitwise_not(ret_mask)
    else:
        if white_proportion < 0.5:
            ret_mask = cv2.bitwise_not(ret_mask)
    return ret_mask


def postprocess_steps(src,dst,class_name):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # closing
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = check_and_invert_mask(img, background=False)
    img[img<128] = 0
    img[img>=128] = 255
    # img[img>0] = 255
    img = np.stack([img,img,img],axis=2)
    cv2.imwrite(dst, img)
    return



groundingdino_model = load_model_hf(CKPT_REPO_ID, CKPT_FILENAME, CKPT_CONFIG_FILENAME, device)
sam_predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device))
groundingdino_model.eval()

for img_path in img_paths:
    gsam(input_path=img_path, output_path=output_dir, class_name="", box_threshold=box_thres, groundingdino_model=groundingdino_model, sam_predictor=sam_predictor, is_invert=True)
    postprocess_steps(output_dir+os.path.basename(img_path), output_dir+os.path.basename(img_path), "")

    inp = "/home/tokichan/research/i-patchcore/src/segment/output/" + os.path.basename(img_path)
    out = "/home/tokichan/research/i-patchcore/src/segment/" + os.path.basename(img_path)

    img = cv2.imread(inp)
    mask = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_id = 0
    output = {}
    for contour in contours:
        instance_area = cv2.contourArea(contour)
        print(instance_area)

        ### bitwise and
        if instance_area > 10:
            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)
            img = cv2.bitwise_and(img, img, mask=new_mask)
            cv2.imwrite(out, img)