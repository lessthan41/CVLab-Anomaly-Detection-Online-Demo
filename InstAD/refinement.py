import numpy as np
import cv2
import os
import random
import tqdm
import glob
import argparse
import PIL.Image as Image
import torchvision.transforms as T
import random
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from .segment.utils import *
from .segment.cnn.dataset import ImageDataset, InstanceDataset, image_transform, update_instances
from .segment.cnn.backbones import load
from .segment.cnn.feature_extractor import YuShuanPatch
from .segment.cnn.common import NetworkFeatureAggregator
from .segment.cnn.utils import kl_divergence

from segment_anything import sam_model_registry, SamPredictor

from .segment.sam import segment_anything
from .segment.gsam import load_model_hf

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_reference_path(output_path, class_name):
    _path = os.path.join(output_path,class_name,"test","good")
    _ref_dir = os.listdir(_path)
    _ref_dir.sort()
    ref_path = os.path.join(_path,_ref_dir[0])
    return ref_path

def generate_instances(img_path="", mask_path="", mask_filename="", inst_filename="", pos_filename="", inst_thre=0.01, target_size=256, refine=False, save_masks=True):
    """ Given img/mask path, output instances/mask/position"""
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path) if save_masks else mask_path
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    mask = cv2.Canny(mask, 20, 160)
    mask = dilate(mask,ksize=3)
    total_area = get_mask_area(mask) / 255
    min_instance_area = inst_thre*total_area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_id = 0
    output = {}
    for contour in contours:
        instance_area = cv2.contourArea(contour)
        if instance_area < min_instance_area:
            continue
        ### YUSHUAN
        x, y, w, h = cv2.boundingRect(contour)
        if w > mask.shape[1]*0.97:
            continue
        diagonal = np.sqrt(w**2+h**2)
        scale = target_size / diagonal
        cropped_mask = np.zeros_like(mask)
        cv2.drawContours(cropped_mask, [contour], -1, 255, thickness=cv2.FILLED)
        cropped_mask = cropped_mask[y:y+h, x:x+w]
        cropped_mask = np.stack([cropped_mask,cropped_mask,cropped_mask],axis=2)
        cropped_image = image[y:y+h, x:x+w]

        temp_size = (np.array(cropped_mask.shape) * scale).astype(np.int32)
        if temp_size[0]==0 or temp_size[1]==0:
            continue
        cropped_mask = cv2.resize(cropped_mask,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
        cropped_image = cv2.resize(cropped_image,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
        cropped_mask[cropped_mask>128] = 255
        cropped_mask[cropped_mask<=128] = 0
        
        # padding
        padw = max(int((target_size - cropped_mask.shape[1])//2), 0)
        padh = max(int((target_size - cropped_mask.shape[0])//2), 0)
        cropped_mask = cv2.copyMakeBorder(cropped_mask,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))
        cropped_image = cv2.copyMakeBorder(cropped_image,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))

        if cropped_mask.shape[0] != target_size:
            cropped_mask = cv2.copyMakeBorder(cropped_mask,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
            cropped_image = cv2.copyMakeBorder(cropped_image,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
        if cropped_mask.shape[1] != target_size:
            cropped_mask = cv2.copyMakeBorder(cropped_mask,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))
            cropped_image = cv2.copyMakeBorder(cropped_image,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))
        
        center_position = [(x+w/2)/mask.shape[1], (y+h/2)/mask.shape[0]]
    
        # crop image by cropped_mask
        cropped_image = crop_by_mask(cropped_image, cropped_mask)

        # output file
        def generate_instance_id(inst_path, img_id, contour_id, refine=False):
            if refine is False:
                return contour_id
            curr = glob.glob(os.path.join(inst_path, f"*{img_id}*"))
            curr = [os.path.basename(i).split("_")[-1][:-4] for i in curr]
            curr = [int(i) for i in curr]
            curr.sort()
            cnt = 0
            for i in curr:
                if i!=cnt:
                    return cnt
                cnt += 1
            return cnt
        
        inst_path = os.path.dirname(inst_filename)
        img_id = os.path.basename(inst_filename)[:-4]
        inst_idx = generate_instance_id(inst_path, img_id, contour_id, refine=refine)
        m_fname = os.path.join(
            mask_filename[:-4]+"_"+str(inst_idx)+mask_filename[-4:]
        )
        i_fname = os.path.join(
            inst_filename[:-4]+"_"+str(inst_idx)+inst_filename[-4:]
        )
        cv2.imwrite(m_fname, cropped_mask)
        cv2.imwrite(i_fname, cropped_image)
        
        # print(center_position,scale)
        mask_name = os.path.basename(inst_filename)
        output[mask_name[:-4]+"_"+str(inst_idx)+mask_name[-4:]] = {
            "center": center_position,
            "scale": scale
        }
        contour_id += 1
    
    ### output position information
    if refine is True:
        curr_pos = load_pickle(pos_filename)
        curr_pos = curr_pos | output
        save_pickle(pos_filename, curr_pos)
    else:
        save_pickle(pos_filename, output)
    return

def instance_segment(class_name,img_input,mask_input,img_output,mask_output,pos_output,args,mode="train"):
    """
        output instance mask, instance image and position {center, scale}
    """
    if mode=="train":
        defect_types = ["good"]
    else:
        defect_types = os.listdir(os.path.join(mask_input,class_name,mode))
    ### instance mask generating
    for defect_type in defect_types:
        print(f"'{defect_type}' Instance Mask generating...")
        mask_input_path = os.path.join(mask_input,class_name,mode,defect_type)
        mask_output_path = os.path.join(mask_output,class_name,mode,defect_type)
        img_output_path = os.path.join(img_output,class_name,mode,defect_type)
        pos_output_path = os.path.join(pos_output,class_name,mode,defect_type)
        os.makedirs(mask_output_path, exist_ok=True)
        os.makedirs(img_output_path, exist_ok=True)
        os.makedirs(pos_output_path, exist_ok=True)
        for mask_name in tqdm.tqdm(os.listdir(mask_input_path)):
            if os.path.exists(os.path.join(mask_output_path,mask_name[:-4]+"_0"+mask_name[-4:])):
                continue
            generate_instances(
                img_path=os.path.join(img_input,class_name,mode,defect_type,mask_name),
                mask_path=os.path.join(mask_input_path,mask_name),
                mask_filename=os.path.join(mask_output_path,mask_name),
                inst_filename=os.path.join(img_output_path,mask_name),
                pos_filename=os.path.join(pos_output_path, mask_name[:-4]+".pkl"),
                target_size=args.target_size,
            )
    return

def instance_feature_extraction(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, mask_path, device, compute_mean=True):
    _instances = ImageDataset(imagesize=target_size, img_id=img_id, instance_dirs=instance_dirs, interval=interval)
    dataloader = torch.utils.data.DataLoader(
        _instances,
        batch_size=inst_batch_size,
        shuffle=False,
    )
    all_feats = []
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        feature = backbone(image)
        all_feats.append(feature.cpu())
    
    all_feats = torch.cat(all_feats, dim=0)
    bundle = 360 // interval
    mean_feats = all_feats.view(all_feats.shape[0] // bundle, bundle, -1)
    mean_feats = mean_feats.mean(dim=1) if compute_mean else mean_feats
    # mean_feats = mean_feats.max(dim=1)[0] if compute_mean else mean_feats ### use max instead
    return mean_feats

def update_mean_feats(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, mask_path, device):
    mean_feats = []
    for img in img_id:
        _ = instance_feature_extraction(backbone, img, instance_dirs, interval, target_size, inst_batch_size, mask_path, device)
        mean_feats.append(_)
    mean_feats = torch.cat(mean_feats, dim=0)
    mean_feats = F.normalize(mean_feats, dim=1)
    return mean_feats.cpu().detach().numpy()

def update_instance_areas(instances_mask, pos):
    """compute areas of binary masks"""
    areas = []
    for mask_name in instances_mask:
        mask = cv2.imread(mask_name) 
        areas.append(get_mask_area(mask) / pos[os.path.basename(mask_name)]["scale"])
    return np.array(areas)

def tsne(mean_feats, instances, img_path, output_filename, cluster_labels, iter, output_path="./"):
    ### tsne
    if len(instances) <= 5:
        return
    p = 10 if len(instances) > 10 else 5
    tsne = TSNE(n_components=2, perplexity=p)
    embedded_data = tsne.fit_transform(mean_feats)

    ### 2d TSNE
    # Plot t-SNE with images for each cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # for cluster in range(n): # kmeans
    for cluster in np.unique(cluster_labels): # meanshift
        plt.scatter(embedded_data[cluster_labels == cluster, 0], 
                    embedded_data[cluster_labels == cluster, 1],
                    label=f'Cluster {cluster + 1}')
        # plot images
        cluster_points = embedded_data[cluster_labels == cluster]
        images = np.array(instances)[cluster_labels == cluster]
        for i, point in enumerate(cluster_points):
            img = cv2.imread(images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (144, 144))
            imagebox = OffsetImage(img, zoom=0.15)
            ab = AnnotationBbox(imagebox, point,
                                xybox=(20,20),  # Adjust the position of the image relative to the point
                                xycoords='data',
                                boxcoords="offset points")
            plt.gca().add_artist(ab)
    # ax.set_title('t-SNE Plot with K-means Clustering (k=2)')
    ax.set_title(f"t-SNE Plot of iterations={iter}")
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True)

    _ = img_path.split("/")
    path = ""
    for s in _[-3:]:
        path += s + "/"
    os.makedirs(os.path.join(output_path, path), exist_ok=True)
    plt.savefig(os.path.join(output_path, path, output_filename))
    plt.close()
    return

def compute_cluster_mean_area(masks, pos):
    """compute mean area of binary masks with rescaling to original size"""
    area = 0
    for mask_name in masks:
        mask = cv2.imread(mask_name)
        area += get_mask_area(mask) / pos[os.path.basename(mask_name)]["scale"]
    return area / len(masks)

def load_pos(pos_dir, img_id):
    """load position information for all instances in img_id"""
    pos = {}
    for i, img in enumerate(img_id):
        pos_path = os.path.join(pos_dir, img+".pkl")
        _ = load_pickle(pos_path)
        pos = pos | _
    return pos

def remove_instances(inst_list):
    """remove failed instances"""
    img_set = set()
    for inst in inst_list:
        img_set.add(convert_to_img_path(inst))

        inst_path = inst
        mask_path = inst.replace("instance","instance_mask")
        pos_path = convert_to_pos_path(inst)
        
        os.remove(inst_path)
        os.remove(mask_path)
        pos = load_pickle(pos_path)
        pos.pop(os.path.basename(inst))
        save_pickle(pos_path, pos)
    
    ### reorder instances
    for img_path in img_set:
        mask_path = img_path.replace("instance","instance_mask")
        pos_path = convert_to_pos_path(img_path)
        pos = load_pickle(pos_path)
        inst_id = [int(os.path.basename(i).split("_")[-1][:-4]) for i in glob.glob(img_path[:-4]+"_*"+img_path[-4:])]
        inst_id.sort()
        idx = 0
        for i in inst_id:
            while i > idx:
                os.rename(img_path[:-4]+"_"+str(inst_id[-1])+img_path[-4:], img_path[:-4]+"_"+str(idx)+img_path[-4:])
                os.rename(mask_path[:-4]+"_"+str(inst_id[-1])+mask_path[-4:], mask_path[:-4]+"_"+str(idx)+mask_path[-4:])
                pos[os.path.basename(img_path[:-4]+"_"+str(idx)+img_path[-4:])] = pos.pop(os.path.basename(img_path[:-4]+"_"+str(inst_id[-1])+img_path[-4:]))
                inst_id.pop()
                idx += 1
            idx += 1
                
        save_pickle(pos_path, pos)
    return

def resegment(cluster_labels, class_name, image_path, instances, pos, mean_inst_area, mean_feats, target_size, working_directory, device, sam_model_type, iter, sam=None, amg_kwargs=None):
    print("Refining segment...")
    ### refine masks with cluster_labels!=0
    inst_to_be_removed = []
    inst_paths = np.array(instances)[cluster_labels!=0]
    for i, inst_path in enumerate(tqdm.tqdm(inst_paths)):
        mask_path = inst_path.replace("instance","instance_mask")
        inst_pos = pos[os.path.basename(inst_path)]
        orig_img = cv2.imread(image_path)

        ### crop out instance
        crop_img, _border = crop_instance(orig_img, inst_pos, target_size)
        y1,y2,x1,x2 = _border
        
        ### mask crop_img using original inst_mask
        gsam_mask = cv2.imread(mask_path)[:,:,0]
        gsam_mask = cv2.resize(gsam_mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        gsam_mask[gsam_mask>128] = 255
        gsam_mask[gsam_mask<=128] = 0
        crop_img = crop_by_mask(crop_img, gsam_mask)
        os.makedirs(f"{working_directory}/output", exist_ok=True)
        cv2.imwrite(f"{working_directory}/output/{class_name}.png", crop_img)

        ### sam
        color_mask, masks = segment_anything(
            sam=sam,
            input=f"{working_directory}/output/{class_name}.png",
            output=f"{working_directory}/output/", 
            model_type=sam_model_type,
            device=device, 
            log=False,
            amg_kwargs=amg_kwargs,
        )
        masks = remove_duplicate_mask(masks, gsam_mask)
        
        ### find instances
        inst_count = 0
        while len(masks) > 0:
            new_inst_mask = np.zeros_like(gsam_mask)
            ### iterate over masks and check overlap ratio
            remove_idxs = []
            for m, mask in enumerate(masks):
                mask_area = get_mask_area(mask)
                new_inst_area = np.sum(new_inst_mask)
                gsam_overlap = np.logical_and(gsam_mask, mask)
                if gsam_overlap.sum() > 0.8*mask_area:
                    mask_overlap = np.logical_and(new_inst_mask, mask)
                    _ = mask_overlap.sum()
                    ### output mask
                    if _ > 0.9*mask_area or _ > 0.9*new_inst_area or new_inst_area == 0:
                        new_inst_mask = np.logical_or(new_inst_mask, mask)
                        remove_idxs.append(m)
                else:
                    remove_idxs.append(m)
            
            ### remove masks in remove_idxs
            masks = np.delete(masks, remove_idxs, axis=0)

            ### check mask size is available
            if np.sum(new_inst_mask)/inst_pos["scale"] < 0.5*mean_inst_area:
                continue ### discard mask

            ### save mask
            new_inst_mask = new_inst_mask.astype(np.uint8)*255
            canvas = np.zeros_like(orig_img)[:,:,0]
            new_inst_mask = fit_mask(new_inst_mask, canvas.shape, (y1,y2,x1,x2))
            canvas[y1:y2,x1:x2] = new_inst_mask
            cv2.imwrite(f"{working_directory}/output/{class_name}.png", canvas)
            _ = convert_to_img_path(inst_path)
            generate_instances(
                img_path=image_path,
                mask_path=f"{working_directory}/output/{class_name}.png",
                mask_filename=_.replace("instance","instance_mask"),
                inst_filename=_,
                pos_filename=_.replace("instance","position")[:-4]+".pkl",
                refine=True,
                target_size=target_size,
            )
            inst_count += 1

        
        gsam_area = get_mask_area(gsam_mask)
        if inst_count != 0:
            inst_to_be_removed.append(inst_path)
        elif iter > 2 and gsam_area/inst_pos["scale"] < 0.2*mean_inst_area:
            inst_to_be_removed.append(inst_path)

    remove_instances(inst_to_be_removed)
    return

def refine_cluster_by_area(cluster_labels, instances, instances_mask, pos, mean_inst_area, threshold=0.2):
    """refine cluster using mask area threshold"""    
    ### judge masks with cluster_labels==0
    indice = [i for i, value in enumerate(cluster_labels) if value==0]
    masks = np.array(instances_mask)[cluster_labels==0]
    for i, mask_name in enumerate(masks):
        mask = cv2.imread(mask_name)
        mask_area = get_mask_area(mask) / pos[os.path.basename(mask_name)]["scale"]
        if abs(mask_area-mean_inst_area)/mean_inst_area > threshold:
            cluster_labels[indice[i]] = 1
    return

def refine_cluster_by_feature(cluster_labels, mean_feats, threshold=0.8):
    """refine cluster using features"""
    normal_feat = mean_feats[cluster_labels==0].mean(axis=0)
    for i, feat in enumerate(mean_feats):
        if cluster_labels[i] != 0:
            continue
        if F.cosine_similarity(torch.from_numpy(feat).unsqueeze(0), torch.from_numpy(normal_feat).unsqueeze(0)).item() < threshold:
            cluster_labels[i] = 1
    return

def refine(class_name, backbone, instance_dirs, image_path, mask_path, img_id, interval, target_size, working_directory, sam_model_type, inst_batch_size, device, refine_step=0.3, sam=None, amg_kwargs=None, pos_path="", output_filename=""):
    ### update instances / mean_feats / instances_mask / pos
    instances = update_instances(img_id, instance_dirs)
    mean_feats = update_mean_feats(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, mask_path, device)
    instances_mask = [i.replace("instance","instance_mask") for i in instances]
    # pos_path = mask_input_path.replace("gsam_sam_intersect", "position")
    pos = load_pos(pos_path, img_id)
    areas = update_instance_areas(instances_mask, pos)
    max_area = areas.max(); min_area = areas.min()
    areas = (areas - min_area) / (max_area - min_area) if max_area != min_area else areas

    ### mean shift
    cluster_labels, cluster_centers = detect_outliers(areas, b=2)

    ### compute mean area of binary masks with all cluster_labels=0
    masks = np.array(instances_mask)[cluster_labels==0]
    mean_inst_area = compute_cluster_mean_area(masks, pos)
    
    ### tsne
    if False:
        tsne(mean_feats, instances, mask_input_path, output_filename, cluster_labels, iter=0, output_path=f"{working_directory}/output/iter=0")

    iter=1
    prev_cluster_labels = np.array([])
    while len(np.unique(cluster_labels)) > 1 and not np.array_equal(prev_cluster_labels, cluster_labels):
        ### update prev_cluster_labels
        prev_cluster_labels = cluster_labels.copy()
        
        ### refine segment
        resegment(cluster_labels, class_name, image_path, instances, pos, mean_inst_area, mean_feats, target_size, working_directory, device, sam_model_type, sam=sam, amg_kwargs=amg_kwargs, iter=iter)

        ### update instances / mean_feats / instances_mask / pos
        instances = update_instances(img_id, instance_dirs)
        mean_feats = update_mean_feats(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, mask_path, device)
        instances_mask = [i.replace("instance","instance_mask") for i in instances]
        pos = load_pos(pos_path, img_id)
        areas = update_instance_areas(instances_mask, pos)
        max_area = areas.max(); min_area = areas.min()
        areas = (areas - min_area) / (max_area - min_area) if max_area != min_area else areas

        ### update mean shift
        cluster_labels, cluster_centers = detect_outliers(areas, b=(2-refine_step*iter))

        ### compute mean area of binary masks with all cluster_labels=0
        masks = np.array(instances_mask)[cluster_labels==0]
        mean_inst_area = compute_cluster_mean_area(masks, pos)

        ### tsne
        if False:
            tsne(mean_feats, instances, mask_input_path, output_filename, cluster_labels, iter=iter, output_path=f"{working_directory}/output/iter={iter}")
        iter += 1
    
    ### update instances / mean_feats / instances_mask / pos
    instances = update_instances(img_id, instance_dirs)
    mean_feats = update_mean_feats(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, mask_path, device)
    instances_mask = [i.replace("instance","instance_mask") for i in instances]
    pos = load_pos(pos_path, img_id)
    areas = update_instance_areas(instances_mask, pos)
    areas = (areas - areas.min()) / (areas.max() - areas.min()) if areas.max() != areas.min() else areas

    ### update mean shift
    cluster_labels, cluster_centers = detect_outliers(areas, b=2)
    masks = np.array(instances_mask)[cluster_labels==0]
    mean_inst_area = compute_cluster_mean_area(masks, pos)

    resegment(cluster_labels, class_name, image_path, instances, pos, mean_inst_area, mean_feats, target_size, working_directory, device, sam_model_type, sam=sam, amg_kwargs=amg_kwargs, iter=iter)

    ### tsne
    if False:
        ### update instances / mean_feats / instances_mask / pos
        instances = update_instances(img_id, instance_dirs)
        mean_feats = update_mean_feats(backbone, img_id, instance_dirs, interval, target_size, inst_batch_size, device)
        instances_mask = [i.replace("instance","instance_mask") for i in instances]
        pos = load_pos(pos_path, img_id)
        areas = update_instance_areas(instances_mask, pos)
        max_area = areas.max(); min_area = areas.min()
        areas = (areas - min_area) / (max_area - min_area) if max_area != min_area else areas

        ### update mean shift
        cluster_labels, cluster_centers = detect_outliers(areas, b=2)

        ### compute mean area of binary masks with all cluster_labels=0
        masks = np.array(instances_mask)[cluster_labels==0]
        mean_inst_area = compute_cluster_mean_area(masks, pos)
        tsne(mean_feats, instances, mask_input_path, output_filename, cluster_labels, iter=iter, output_path=f"{working_directory}/output/iter={iter}")

    return

def load_backbone(backbone="resnet18", layers_to_extract_from=["layers2","layers3"], imagesize=256, device="cuda"):
    backbone = load(backbone)
    backbone.to(device)
    backbone.eval()
    feature_aggregator = NetworkFeatureAggregator(
        backbone, layers_to_extract_from, imagesize=imagesize, device=device, train_backbone=False
    )
    _ = feature_aggregator.eval()
    return feature_aggregator

def load_sam(device="cuda", sam_checkpoint="/home/anomaly/sam_ckpt/sam_vit_b_01ec64.pth"):
    model_type = "vit_h" if "vit_h" in args.sam_checkpoint else "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.eval()
    _ = sam.to(device=device)
    amg_kwargs = {
        "points_per_side": 16,
        "points_per_batch": None,
        "pred_iou_thresh": None,
        "stability_score_thresh": 0.9,
        "stability_score_offset": None,
        "box_nms_thresh": 0.9,
        "crop_n_layers": 1,
        "crop_nms_thresh": 0.9,
        "crop_overlap_ratio": None,
        "crop_n_points_downscale_factor": None,
        "min_mask_region_area": None,
    }
    return sam, amg_kwargs

def load_gsam(device="cuda", args=None, sam_checkpoint="/home/anomaly/sam_ckpt/sam_vit_b_01ec64.pth"):
    model_type = "vit_h" if "vit_h" in args.sam_checkpoint else "vit_b"
    groundingdino_model = load_model_hf(args.ckpt_repo_id, args.ckpt_filename, args.ckpt_config_filename, device)
    sam_predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device))
    groundingdino_model.eval()
    return groundingdino_model, sam_predictor

def refine_segment(class_name,image_path,mask_path,instance_path,target_size,sam_checkpoint,device,interval=30,mode="train"):
    """extract feature from instance"""
    backbone = YuShuanPatch(backbone_name="wide_resnet50_2", imagesize=target_size, device=device, normalize=True)
    sam, amg_kwargs = load_sam(sam_checkpoint=sam_checkpoint, device=device)
    defect_types = os.listdir(os.path.join(instance_path,class_name,mode))
    for defect_type in defect_types:
        print(f"'{defect_type}' Feature Extracting...")
        imgs = os.listdir(os.path.join(mask_path,class_name,mode,defect_type))
        imgs = [os.path.join(mask_path,class_name,mode,defect_type,img) for img in imgs]
        imgs.sort()
        instance_dirs = set()
        instance_dirs.add(os.path.join(instance_path,class_name,mode,defect_type))

        inst_cnt = 0
        img_id = []
        for i, img in enumerate(tqdm.tqdm(imgs)):
            img_id.append(os.path.basename(img)[:-4])
            inst_cnt = get_instances_number(img_id, instance_dirs)
            if inst_cnt >= args.cluster_batch_size or i==len(imgs)-1:
                output_filename = link_string(img_id) + ".png"
                refine(
                    class_name=class_name,
                    backbone=backbone,
                    sam=sam,
                    amg_kwargs=amg_kwargs,
                    instance_dirs=instance_dirs,
                    image_path=image_path,
                    mask_input_path=os.path.dirname(img), 
                    img_id=img_id,
                    output_filename=output_filename,
                    interval=interval,
                    args=args,
                )
                
                ### reset counter
                inst_cnt = 0
                img_id = []

    return

def refine_segment_demo(image_path,class_name,data_path,pos_path,mask_path,instance_path,output_path,target_size,batch_size,refine_step,sam,sam_model_type,backbone,amg_kwargs,device,interval=30):
    """extract feature from instance"""
    print(f"Refining Instances...")
    instance_dirs = set()
    instance_dirs.add(instance_path)
    img_id = [os.path.basename(image_path)[:-4]]
    refine(
        class_name=class_name,
        backbone=backbone,
        sam=sam,
        amg_kwargs=amg_kwargs,
        instance_dirs=instance_dirs,
        image_path=image_path,
        pos_path=pos_path,
        img_id=img_id,
        output_filename="",
        interval=interval,
        target_size=target_size,
        working_directory=output_path,
        mask_path=mask_path,
        device=device,
        inst_batch_size=batch_size,
        refine_step=refine_step,
        sam_model_type=sam_model_type,
    )
    return

def feature_alignment(backbone,class_name,ref_path,instance_path,mask_path,position_path,target_size,inst_batch_size,interval=30,mode="train",device="cuda"):
    """align instances"""
    # backbone = load_backbone("resnet18", ["layer2","layer3"], imagesize=args.target_size, device=args.device)

    ### construct reference feature
    ref_instance = Image.open(ref_path).convert("RGB")
    ref_instance = image_transform(ref_instance, target_size).unsqueeze(0).to(device)
    # ref_feature = backbone.feature_extraction(ref_instance)
    ref_feature = backbone(ref_instance)
    
    ### iterate over all instances
    defect_types = os.listdir(os.path.join(instance_path,class_name,mode))
    for defect_type in defect_types:
        print(f"'{defect_type}' Instance Angle Aligning...")
        instances = os.listdir(os.path.join(instance_path,class_name,mode,defect_type))
        instances.sort()
        prev_img_id = instances[0].split("_")[0]
        _ = os.path.join(position_path,class_name,mode,defect_type,instances[0].split("_")[0]+".pkl")
        pos_information = load_pickle(_)
        for i in tqdm.tqdm(instances):
            img_id = i.split("_")[0]
            ### single image quick pass
            # if img_id not in ["057","058"]:
            #     continue
            ###
            ### update position information
            if img_id != prev_img_id:
                _ = os.path.join(position_path,class_name,mode,defect_type,prev_img_id+".pkl")
                save_pickle(_,pos_information)
                _ = os.path.join(position_path,class_name,mode,defect_type,img_id+".pkl")
                pos_information = load_pickle(_)
                prev_img_id = img_id

            ### image-level align
            score_list = []
            i_path = os.path.join(instance_path,class_name,mode,defect_type,i)
            _instances = InstanceDataset(
                imagesize=target_size, 
                i_path=i_path, 
                interval=interval
            )
            dataloader = torch.utils.data.DataLoader(
                _instances,
                batch_size=inst_batch_size,
                shuffle=False,
            )
            for _i, batch in enumerate(dataloader):
                image = batch["image"].to(device)
                inst_feature = backbone(image)
                # sim = F.cosine_similarity(ref_feature, inst_feature).tolist()
                sim = kl_divergence(ref_feature, inst_feature)
                score_list += sim

            best_angle = np.argmin(score_list) * interval

            ### save rotated instance and information
            instance = Image.open(i_path).convert("RGB")
            instance = instance.rotate(best_angle, resample=Image.BILINEAR)
            instance.save(i_path)
            mask = Image.open(i_path.replace(instance_path, mask_path)).convert("RGB")
            mask = mask.rotate(best_angle, resample=Image.BILINEAR)
            mask.save(i_path.replace(instance_path, mask_path))
            if "angle" in pos_information[i].keys(): ### exist key=angle
                pos_information[i]["angle"] += best_angle
                pos_information[i]["angle"] = pos_information[i]["angle"] % 360
            else:

                pos_information[i]["angle"] = best_angle
        
        ### update position information
        _ = os.path.join(position_path,class_name,mode,defect_type,img_id+".pkl")
        save_pickle(_,pos_information)

    return

def feature_alignment_demo(image_path,backbone,class_name,ref_path,instance_path,mask_path,position_path,target_size,inst_batch_size,interval=30,device="cuda"):
    """align instances"""
    ### construct reference feature
    ref_instance = Image.open(ref_path).convert("RGB")
    ref_instance = image_transform(ref_instance, target_size).unsqueeze(0).to(device)
    ref_feature = backbone(ref_instance)

    ### iterate over instances
    instances = glob.glob(f"{instance_path}/*.*")
    position = glob.glob(f"{position_path}/*.*")[0]
    position = load_pickle(position)

    ### single image
    for i in tqdm.tqdm(instances):
        score_list = []
        _instances = InstanceDataset(
            imagesize=target_size, 
            i_path=i, 
            interval=interval
        )
        dataloader = torch.utils.data.DataLoader(
            _instances,
            batch_size=inst_batch_size,
            shuffle=False,
        )
        for _i, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            inst_feature = backbone(image)
            # sim = F.cosine_similarity(ref_feature, inst_feature).tolist()
            sim = kl_divergence(ref_feature, inst_feature)
            score_list += sim

        best_angle = np.argmin(score_list) * interval

        ### save rotated instance and information
        instance = Image.open(i).convert("RGB")
        instance = instance.rotate(best_angle, resample=Image.BILINEAR)
        instance.save(i)
        mask = Image.open(i.replace(instance_path, mask_path)).convert("RGB")
        mask = mask.rotate(best_angle, resample=Image.BILINEAR)
        mask.save(i.replace(instance_path, mask_path))
        if "angle" in position[os.path.basename(i)].keys():
            position[os.path.basename(i)]["angle"] += best_angle
            position[os.path.basename(i)] = position[os.path.basename(i)]["angle"] % 360
        else:
            position[os.path.basename(i)]["angle"] = best_angle


    ### update position information
    save_pickle(f"{position_path}/{os.path.basename(image_path)[:-4]}.pkl",position)

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument('-c', '--classes', type=str, help='classes for segmentation', default='macaroni1')
    # parser.add_argument('-c', '--classes', type=str, help='classes for segmentation', default='tubes')
    parser.add_argument('-d', '--dataset', type=str, help='dataset', default='visa')
    parser.add_argument('-s', '--target_size', type=int, help='target size', default=256)
    parser.add_argument('--inst_batch_size', type=int, help='instance batch size', default=8)
    parser.add_argument('--cluster_batch_size', type=int, help='batch size', default=20)
    parser.add_argument('--output_path', type=str, help='output path', default='/home/anomaly/data/segment/output')
    # parser.add_argument('--data_path', type=str, help='data path', default='/home/anomaly/data/MPDD')
    parser.add_argument('--data_path', type=str, help='data path', default='/home/anomaly/data/VisA_highshot')
    parser.add_argument('--working_directory', type=str, help='working directory path', default='/home/anomaly/research/i-patchcore/src/segment/')
    parser.add_argument("--ckpt_repo_id", type=str, help="checkpoint repo id", default="ShilongLiu/GroundingDINO")
    parser.add_argument("--ckpt_filename", type=str, help="checkpoint filename", default="/home/anomaly/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--ckpt_config_filename", type=str, help="checkpoint config filename", default="/home/anomaly/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--sam_checkpoint', type=str, help='sam checkpoint', default='/home/anomaly/data/ckpt/sam_vit_b_01ec64.pth')
    # parser.add_argument('--sam_checkpoint', type=str, help='sam checkpoint', default='/home/anomaly/data/ckpt/sam_vit_h_4b8939.pth')
    parser.add_argument('--device', type=str, help='device', default='cuda')
    parser.add_argument('--refine_segment', action='store_true', help='refine segment', default=False)
    parser.add_argument('--feature_alignment', action='store_true', help='feature alignment', default=False)
    parser.add_argument('--train', action='store_true', help='training set postprocess', default=False)
    parser.add_argument('--test', action='store_true', help='testing set postprocess', default=False)
    parser.add_argument('--reset', action='store_true', help='rerun everything (False for resume)', default=False)
    parser.add_argument('--demo', action='store_true', help='if True, random sample 10 images and output tsne', default=False)
    args = parser.parse_args()

    train_img_input = f"{args.output_path}/{args.dataset}/image/"
    data_path = f"{args.data_path}"
    mask_input = f"{args.output_path}/{args.dataset}/gsam_sam_intersect/"
    img_output = f"{args.output_path}/{args.dataset}/instance/"
    mask_output = f"{args.output_path}/{args.dataset}/instance_mask/"
    position_output = f"{args.output_path}/{args.dataset}/position/"
    classes = os.listdir(train_img_input)
    
    print("Instance segmenting...")
    for class_name in classes:
        if class_name not in args.classes.split(","):
            continue
        print(class_name)
        if args.train:
            if args.reset:
                reset_folder(os.path.join(mask_output, class_name, "train"))
                reset_folder(os.path.join(img_output, class_name, "train"))
                reset_folder(os.path.join(position_output, class_name, "train"))
            instance_segment(class_name,train_img_input,mask_input,img_output,mask_output,position_output,args,mode="train")
            if args.refine_segment:
                refine_segment(class_name,data_path,mask_input,img_output,args.target_size,args.sam_checkpoint,args.device,interval=30,mode="train")
            if args.feature_alignment:
                ref_path = get_reference_path(img_output, class_name)
                feature_alignment(class_name,ref_path,img_output,mask_output,position_output,args,interval=10,mode="train")
        if args.test:
            if args.reset:
                reset_folder(os.path.join(mask_output, class_name, "test"))
                reset_folder(os.path.join(img_output, class_name, "test"))
                reset_folder(os.path.join(position_output, class_name, "test"))
            instance_segment(class_name,data_path,mask_input,img_output,mask_output,position_output,args,mode="test")
            if args.refine_segment:
                refine_segment(class_name,data_path,mask_input,img_output,args.target_size,args.sam_checkpoint,args.device,interval=30,mode="test")
            if args.feature_alignment:
                ref_path = get_reference_path(img_output, class_name)
                feature_alignment(class_name,ref_path,img_output,mask_output,position_output,args,interval=10,mode="test")


    