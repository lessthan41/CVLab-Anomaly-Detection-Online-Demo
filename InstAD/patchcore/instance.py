import numpy as np
import os
import glob
import cv2
import pickle
from scipy.ndimage import rotate

def paste_instance(score, segmentation, position, img_path, input_shape, resize_size=256):
    ### construct score map
    orig_score = np.copy(segmentation)
    cx,cy = position[os.path.basename(img_path)]["center"]
    scale = position[os.path.basename(img_path)]["scale"]
    if "angle" in position[os.path.basename(img_path)]:
        rotation = position[os.path.basename(img_path)]["angle"]
    else:
        rotation = 0
    coor_cx = int(resize_size*cx)
    coor_cy = int(resize_size*cy)

    diagonal = resize_size/scale
    x = int(resize_size*diagonal/input_shape[1])
    y = int(resize_size*diagonal/input_shape[0])
    orig_score = rotate(orig_score, -rotation, reshape=False)
    orig_score = cv2.resize(orig_score,(x,y))
    
    ### paste score
    anchor_tx = coor_cx - x//2
    anchor_ty = coor_cy - y//2
    anchor_bx = coor_cx - x//2 + x
    anchor_by = coor_cy - y//2 + y

    border = [anchor_tx,anchor_ty,anchor_bx,anchor_by]
    starter = [0,0,x,y]

    if anchor_tx < 0:
        border[0] = 0
        starter[0] = abs(anchor_tx)
    if anchor_ty < 0:
        border[1] = 0
        starter[1] = abs(anchor_ty)
    if anchor_bx > resize_size:
        border[2] = resize_size
        starter[2] = x-(anchor_bx-resize_size)
    if anchor_by > resize_size:
        border[3] = resize_size
        starter[3] = y-(anchor_by-resize_size)
    
    orig_score = orig_score[starter[1]:starter[3],starter[0]:starter[2]]
    score[border[1]:border[3],border[0]:border[2]] = np.maximum(
        score[border[1]:border[3],border[0]:border[2]], orig_score
    )
    return score

def generate_final_anomaly_maps(scores,segmentations,labels_gt,masks_gt,img_paths, data_path, class_name, pos_path):
    ### determine image shape
    img_directory = os.path.join(data_path,class_name,"test","good")
    img_path = glob.glob(os.path.join(img_directory,"*.*"))[0]
    input_shape = cv2.imread(img_path).shape

    ### initialize
    resize_size = segmentations[0].shape[0]
    scores_new = []
    segmentations_new = []
    labels_gt_new = []
    masks_gt_new = []
    scores_instances = []
    labels_gt_instances = []
    prev_defect = os.path.dirname(img_paths[0]).split("/")[-1]
    prev_img_id = os.path.basename(img_paths[0]).split("_")[0]
    score = np.zeros_like(segmentations[0])
    position = None
    for i in range(len(segmentations)):
        ### image id
        img_path = img_paths[i]
        ### BUG
        # pos_path = os.path.join(os.path.dirname(pos_path), os.path.basename(img_path).split("_")[0] + ".pkl") # demo
        pos_path = os.path.join(os.path.dirname(img_path).replace("instance","position"), os.path.basename(img_path).split("_")[0] + ".pkl") # run_patchcore
        ###
        img_id = os.path.basename(img_path).split("_")[0]
        defect = os.path.dirname(img_path).split("/")[-1]
        
        ### read position
        with open(pos_path, "rb") as handler:
            position = pickle.load(handler)

        if img_id!=prev_img_id or defect!=prev_defect:
            # output_score_map
            scores_new.append(max(scores_instances))
            segmentations_new.append(score.tolist())
            labels_gt_new.append(int(any(labels_gt_instances)))
            masks_gt_new.append(masks_gt[i-1])

            # clear record
            score = np.zeros_like(segmentations[0])
            labels_gt_instances = []
            scores_instances = []
        
        score = paste_instance(score, segmentations[i], position, img_path, input_shape, resize_size)
        scores_instances.append(scores[i])
        labels_gt_instances.append(labels_gt[i])

        prev_img_id = img_id
        prev_defect = defect

    # output_score_map
    scores_new.append(max(scores_instances))
    segmentations_new.append(score)
    labels_gt_new.append(any(labels_gt_instances))
    masks_gt_new.append(masks_gt[len(masks_gt)-1])

    return scores_new, segmentations_new, labels_gt_new, masks_gt_new

def output_img(img_paths, scores, segmentations, masks_gt, image_output_path, data_path, dist_metric, class_name):
    x = 0
    for id in range(len(img_paths)):
        directories = os.path.dirname(img_paths[id]).split("/")
        mode = "test"
        defect = directories[-1]
        fname = os.path.basename(img_paths[id]).split('_')[0] + os.path.splitext(img_paths[id])[1]
        # fname = os.path.basename(img_paths[id])[:-4] + str(scores[x]) + os.path.basename(img_paths[id])[-4:]
        directory = f"{image_output_path}/_/{class_name}_{dist_metric}/{mode}/{defect}"
        img_directory = f"{data_path}/{class_name}/{mode}/{defect}/"
        os.makedirs(directory, exist_ok=True)

        if img_paths[id].split("_")[-1][0]=="0":
            orig_img_path = img_directory+fname[:-4] + os.path.splitext(img_paths[id])[1]
            orig_img = cv2.imread(orig_img_path)
            np_segmentations = np.array(segmentations[x])

            score = scores[x]
            new_masks_gt = np.array(masks_gt[x])
            new_masks_gt[new_masks_gt==1] = 255
            new_masks_gt = new_masks_gt.reshape((new_masks_gt.shape[1],new_masks_gt.shape[2],1))
            instance = cv2.imread(img_paths[id])
            np.save(f"{directory}/test_segmentation.npy", np_segmentations)
            cv2.imwrite(f"{directory}/{fname}", orig_img)
            cv2.imwrite(f"{directory}/test_mask_gt.jpg", new_masks_gt)
            cv2.imwrite(f"{directory}/instance.jpg", instance)
            visualize(class_name, defect, fname, score, image_output_path, dist_metric)
            x += 1
    return

def output_img_demo(scores, segmentations, masks_gt, image_output_path, data_path, dist_metric, class_name, image_path):
    x = 0
    directories = os.path.dirname(image_path).split("/")
    mode = "test"
    defect = directories[-1]
    fname = os.path.basename(image_path)
    directory = f"{image_output_path}/_/{class_name}_{dist_metric}/{mode}/{defect}"
    img_directory = f"{data_path}/{class_name}/{mode}/{defect}/"
    os.makedirs(directory, exist_ok=True)

    orig_img = cv2.imread(image_path)
    np_segmentations = np.array(segmentations[0])
    score = scores[0]
    new_masks_gt = np.array(masks_gt[0])
    new_masks_gt[new_masks_gt==1] = 255
    new_masks_gt = new_masks_gt.reshape((new_masks_gt.shape[1],new_masks_gt.shape[2],1))
    np.save(f"{directory}/test_segmentation.npy", np_segmentations)
    cv2.imwrite(f"{directory}/{fname}", orig_img)
    cv2.imwrite(f"{directory}/test_mask_gt.jpg", new_masks_gt)
    visualize(class_name, defect, fname, score, image_output_path, dist_metric)
    return

def visualize(class_name, defect, fname, score, image_output_path, dist_metric, epoch=-1, demo=False):
    if epoch == -1:
        epoch = "final"
    output_path = os.path.join(image_output_path, f"{class_name}_{dist_metric}", str(epoch), defect)
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(image_output_path, "_", f"{class_name}_{dist_metric}", "test", defect)
    
    orig_path = f"{file_path}/{fname}"
    # inst_path = f"{file_path}/instance.jpg"
    seg_path = f"{file_path}/test_segmentation.npy"
    mask_path = f"{file_path}/test_mask_gt.jpg"

    orig_image = cv2.imread(orig_path)
    # inst_image = cv2.imread(inst_path)
    h,w,c = orig_image.shape
    # inst_image = cv2.resize(inst_image,(h,h))

    if "good" in mask_path:
        mask = np.zeros((h,w,c))
    else:
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask,(w,h))
    seg = np.load(seg_path)
    seg = cv2.resize(seg,(w,h))
    if dist_metric == "cosine":
        seg = np.clip(seg * 200, 0, 255).astype(np.uint8)
    elif demo:
        seg = np.clip(seg * 50, 0, 50).astype(np.uint8)
    else:
        seg = np.clip(seg * 255, 0, 255).astype(np.uint8)
    seg = cv2.applyColorMap(seg,cv2.COLORMAP_JET)
    seg = cv2.addWeighted(orig_image, 0.3, seg, 0.7, 0.0)

    result = np.hstack([orig_image,mask,seg]).astype(np.uint8)
    # result = np.hstack([orig_image,inst_image,mask,seg]).astype(np.uint8)
    # print(f"{output_path}/{fname[:-4]}_{str(round(score,2))}{fname[-4:]}")
    cv2.imwrite(f"{output_path}/{fname[:-4]}_{str(round(score,2))}{fname[-4:]}", result)
    return
