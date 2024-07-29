import os
import cv2
import glob
import time
import torch.nn as nn

from .segment.utils import reset_folder
from .segment.sam import segment_anything
from .segment.gsam import groundingdino
from .segment.cnn.feature_extractor import YuShuanPatch
from .instance_segment import load_sam, load_groundingdino, amg_kwargs
from .postprocess import postprocess_steps, intersect_steps
from .refinement import load_backbone, generate_instances, refine_segment_demo, feature_alignment_demo
from .patchcore_module import MyPatchcore
from .config import amg_kwargs

class InstAD(nn.Module):
    def __init__(self, dataset_root, class_name, few_shot=True) -> None:
        super(InstAD, self).__init__()
        self.output_dirs = "/home/tokichan/demo-system/InstAD/instad_output"
        self.device = "cuda"
        self.gpu = [0]
        self.samh_checkpoint = "/home/tokichan/segment-anything/ckpt/sam_vit_h_4b8939.pth"
        self.samb_checkpoint = "/home/tokichan/segment-anything/ckpt/sam_vit_b_01ec64.pth"
        self.ckpt_filename = "/home/tokichan/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.ckpt_config_filename = "/home/tokichan/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.refinement_list = ["capsules","tubes"]

        self.inst_path = os.path.join(self.output_dirs, "instance", f"{class_name}")
        self.mask_path = os.path.join(self.output_dirs, "instance_mask", f"{class_name}")
        self.pos_path = os.path.join(self.output_dirs, "position", f"{class_name}")
        self.dataset_root = dataset_root
        self.class_name = class_name
        self.few_shot = few_shot
        self.batch_size = 8
        self.refine_batch_size = 8
        self.target_size = 256
        self.refine_step = 0.3 ### larger converge faster
        self.refine = True if self.class_name in self.refinement_list else False
        self.sam_h = load_sam(self.samh_checkpoint, "vit_h", self.device)
        self.sam_b = load_sam(self.samb_checkpoint, "vit_b", self.device)
        self.groundingdino, self.sam_predictor = load_groundingdino(self.sam_h, self.ckpt_filename, self.ckpt_config_filename, self.device)
        self.backbone = self.load_backbone()
        self.patchcore = self.load_patchcore()
        self.amg_kwargs = amg_kwargs


    def forward(self, image_path):
        reset_folder(self.output_dirs)
        img_path = self.save_image(image_path)
        ### sam, groundingdino
        sam_mask, gsam_mask = self.segment(img_path)
        ### intersect
        intersect_mask = intersect_steps(gsam_src=gsam_mask, sam_src=sam_mask, threshold=0.95, save_masks=False)
        ### instance segment
        self.instance_segment(img_path, intersect_mask)
        ### refinement
        if self.refine:
            self.refine_instances(img_path)
        ### feature alignment
        self.feature_alignment(img_path)
        ### anomaly detection
        anomaly_score, anomaly_map = self.anomaly_detection(img_path)
        return anomaly_map, anomaly_score
    
    def save_image(self, image_path):
        os.makedirs(self.output_dirs, exist_ok=True)
        img_path = os.path.join(self.output_dirs, f"{os.path.basename(image_path)}").replace("webp", "jpg")
        image = cv2.imread(image_path)
        cv2.imwrite(img_path, image)
        return img_path

    def segment(self, img_path, box_thres=0.5):
        color_mask, sam_mask = segment_anything(self.sam_h, img_path, self.output_dirs, "vit_h", self.device, self.amg_kwargs, save_masks=False)
        gsam_mask = groundingdino(img_path, self.output_dirs, class_name=self.class_name, box_threshold=box_thres, groundingdino_model=self.groundingdino, sam_predictor=self.sam_predictor, is_invert=False, save_masks=False)
        gsam_mask = postprocess_steps(class_name=self.class_name, mask=gsam_mask, save_masks=False)
        return sam_mask, gsam_mask
    
    def instance_segment(self, image_path, intersect_mask):
        reset_folder(self.inst_path)
        reset_folder(self.mask_path)
        reset_folder(self.pos_path)
        generate_instances(
            img_path=image_path,
            mask_path=intersect_mask,
            mask_filename=os.path.join(self.mask_path, f"{os.path.basename(image_path)}"),
            inst_filename=os.path.join(self.inst_path, f"{os.path.basename(image_path)}"),
            pos_filename=os.path.join(self.pos_path, f"{os.path.basename(image_path)[:-4]}.pkl"),
            save_masks=False,
        )
        return

    def refine_instances(self, image_path):
        refine_segment_demo(
            image_path=image_path,
            class_name=self.class_name,
            data_path=self.dataset_root,
            instance_path=self.inst_path,
            pos_path=self.pos_path,
            mask_path=self.mask_path,
            target_size=self.target_size,
            batch_size=self.refine_batch_size,
            refine_step=self.refine_step,
            sam=self.sam_b,
            sam_model_type="vit_b",
            backbone=self.backbone,
            amg_kwargs=self.amg_kwargs,
            output_path=self.output_dirs,
            device=self.device,
        )
        return
    
    def feature_alignment(self, image_path):
        ref_path = glob.glob(f"{self.dataset_root}/{self.class_name}/train/good/*.*")[0]
        feature_alignment_demo(
            image_path=image_path,
            backbone=self.backbone,
            class_name=self.class_name,
            ref_path=ref_path,
            instance_path=self.inst_path,
            mask_path=self.mask_path,
            position_path=self.pos_path,
            target_size=self.target_size,
            inst_batch_size=self.batch_size,
            device=self.device,
        )
        return
    
    def load_patchcore(self):
        # if self.patchcore eixst
        if hasattr(self, "patchcore"):
            del self.patchcore
        patchcore = MyPatchcore(
            inst_path=self.inst_path,
            results_path=self.output_dirs,
            data_path=self.dataset_root,
            pos_path=self.pos_path,
            class_name=self.class_name,
            batch_size=self.batch_size,
            gpu=self.gpu,
            dist_metric="L2",
            zero_shot=not self.few_shot,
            demo=True
        )
        print("successfully load patchcore model================================")
        return patchcore
    
    def load_backbone(self):
        # b = YuShuanPatch(backbone_name="wide_resnet50_2", imagesize=self.target_size, device=self.device, normalize=True)
        b = load_backbone("resnet18", ["layer2","layer3"], imagesize=self.target_size, device=self.device)
        print("successfully load backbone=======================================")
        return b
    
    def anomaly_detection(self, image_path):
        scores, segmentations = self.patchcore.run(image_path=image_path, output_image=False)
        return scores, segmentations

if __name__ == '__main__':

    dataset_root = "/home/tokichan/data/VisA_highshot/"
    class_name = "macaroni2"
    # dataset_root = "/home/tokichan/data/MPDD/"
    # class_name = "tubes"
    
    ### anomaly detection
    instad = InstAD(dataset_root, class_name)
    # anomaly_map, anomaly_score = instad("/home/tokichan/data/VisA_highshot/capsules/test/bubble,discolor/042.JPG")
    
    ### speedtest
    import random
    import numpy as np
    img_list = glob.glob(os.path.join(dataset_root, class_name, "test/*/*"))
    img_list = [i for i in img_list if "good" not in i]
    n = 100
    segment_times = []
    refine_times = []
    feature_alignment_times = []
    ad_times = []
    for i in range(n):
        path = random.choice(img_list)
        time1, time2, time3, time4 = instad(path)
        segment_times.append(time1)
        refine_times.append(time2)
        feature_alignment_times.append(time3)
        ad_times.append(time4)
    times = np.array(segment_times)
    print(f"segment time mean: {times.mean()}, std: {times.std()}")
    times = np.array(refine_times)
    print(f"refine time mean: {times.mean()}, std: {times.std()}")
    times = np.array(feature_alignment_times)
    print(f"feature alignment time mean: {times.mean()}, std: {times.std()}")
    times = np.array(ad_times)
    print(f"anomaly detection time mean: {times.mean()}, std: {times.std()}")

