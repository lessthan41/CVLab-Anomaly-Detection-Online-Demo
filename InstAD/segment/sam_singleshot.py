from segment_anything import sam_model_registry, SamPredictor
from sam import segment_anything
from gsam import gsam, load_model_hf, CKPT_REPO_ID, CKPT_FILENAME, CKPT_CONFIG_FILENAME, SAM_CHECKPOINT
import numpy as np
import cv2

model_type = "vit_h"
device="cuda"


img = cv2.imread("/home/anomaly/data/VisA_highshot/capsules/test/bubble,discolor,scratch/096.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
sam.eval()
_ = sam.to(device=device)

sam_predictor = SamPredictor(sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT).to(device))
sam_predictor.set_image(img)
background,_,_ = sam_predictor.predict(
                                point_coords = np.array([[10,10],[10,img.shape[0]-10],[img.shape[1]-10,10],[img.shape[1]-10,img.shape[0]-10],[0,0],[0,img.shape[0]],[img.shape[1],0],[img.shape[1],img.shape[0]]]),
                                point_labels = np.ones(8),
                                box = None,
                                mask_input = None,
                                multimask_output = True,
                                return_logits = False,)


cv2.imwrite("0.JPG",np.where(background[0], 255, 0).astype(np.uint8))
cv2.imwrite("1.JPG",np.where(background[1], 255, 0).astype(np.uint8))
cv2.imwrite("2.JPG",np.where(background[2], 255, 0).astype(np.uint8))