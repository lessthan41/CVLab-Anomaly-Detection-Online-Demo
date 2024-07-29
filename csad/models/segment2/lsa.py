import cv2
import numpy as np
import glob
import random
import albumentations as A
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from sklearn.metrics import pairwise_distances_argmin_min

color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
        
def turn_binary_to_int(mask):
    temp = np.zeros_like(mask,dtype=np.uint8)
    temp[mask]=255
    return temp

def intersect_ratio(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    if intersection.sum() == 0:
        return 0
    ratio = np.sum(intersection)/min([np.sum(mask1!=0),np.sum(mask2!=0)])
    ratio = 0 if np.isnan(ratio) else ratio
    return ratio

def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.001:
            result_masks.append(mask)
    return result_masks

def sample_point(in_x, in_y,min_distance=0.4,boundary=0.05):
    # Number of points to sample
    assert min_distance < np.sqrt(2*(0.5-boundary)**2), "min_distance should be smaller than np.sqrt(2*(0.5-boundary)**2)"
    num_samples = 10000
    
    # Generate random points
    random_points = np.random.rand(num_samples, 2)
    random_points = random_points*(1-boundary*2) + boundary
    
    # Calculate distances from the input point
    distances = np.sqrt((random_points[:, 0] - in_x)**2 + (random_points[:, 1] - in_y)**2)
    
    # Calculate distances as weights
    distances[distances < min_distance] = 0
    weights = distances**2
    
    
    # Normalize weights to make them probabilities
    probabilities = weights / np.sum(weights)
    
    # Sample a point based on the probabilities
    sampled_index = np.random.choice(num_samples, p=probabilities)
    sampled_point = random_points[sampled_index, :]
    
    return sampled_point
    



        
def sample_mask(masks,weight_power=0.5):
    mask_weight = np.array([np.sum(m) for m in masks])
    mask_weight = mask_weight**weight_power
    # max_mask = np.argmax(mask_weight)
    # mask_weight[max_mask] = 0 # remove the largest mask
    mask_weight = mask_weight/np.sum(mask_weight)
    idx = np.random.choice(np.arange(len(masks)),p=mask_weight)
    source_mask = masks[idx]
    return source_mask, idx

def labeled_lsa(source_img,source_masks,source_labelmap,target_img,target_masks,target_labelmap,target_background,config):
    # both source and target masks are labeled
    # paste source image to target image
    source_masks = split_masks_from_one_mask(source_masks)
    # filter masks that are background
    new_source_masks = list()
    for m in source_masks:
        # cv2.imshow(f"{intersect_ratio(m,source_labelmap)}",np.hstack([m,source_labelmap*30]))
        # cv2.waitKey(0)
        if intersect_ratio(m,source_labelmap) > 0.9:
            new_source_masks.append(m)
    source_masks = new_source_masks

    aug_num = 0
    final_mask = np.zeros_like(target_masks)
    result_img = target_img.copy()
    result_labelmap = target_labelmap.copy()
    while aug_num < config['min_aug_num']:
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        if (source_labelmap[source_mask>0]==0).sum() > 100:
            continue
        for attempt in range(500):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
                if (source_labelmap[source_mask>0]==0).sum() > 100:
                    continue
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,target_background) < 0.5:
                # found a proper position
                # print(f"intersect ratio:{intersect_ratio(result_mask,target_background)}")
                break
        # print(f"attempt:{attempt}")
        if attempt >= 500:
            # failed to find a proper position
            continue
        else:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            aug_num += 1

    # visualize
    # vis_image = np.hstack([source_img,cv2.cvtColor(source_mask,cv2.COLOR_GRAY2BGR),result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)
    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
            
    for i in range(config['max_aug_num']-config['min_aug_num']):
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        if (source_labelmap[source_mask>0]==0).sum() > 100:
            continue
        for attempt in range(500+1):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
                if (source_labelmap[source_mask>0]==0).sum() > 100:
                    continue
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,target_background) < 0.9:
                # found a proper position
                break

        # print(f"attempt:{attempt}")
        if attempt < 500:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            
            aug_num += 1
    ###############################
    # visualize
    # color_labelmap = np.zeros_like(result_img)
    # for i in range(1,np.max(result_labelmap)+1):
    #     color_labelmap[result_labelmap==i] = color_list[i-1][::-1]
    # vis_image = np.hstack([result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR),color_labelmap])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)

    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
    #cv2.imwrite("result.png",vis_image)

    return result_img, result_labelmap#, vis_image

from scipy.interpolate import CubicSpline
def ssa_transform(image, mask, num_points=2, max_translation=50):
    image = image.astype(np.float32)

    # Find contours using OpenCV (consider alternative methods if OpenCV is unavailable)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Validate contour existence (handle cases where no contours are found)
    if not contours:
        print("Warning: No contours found in the mask.")
        return image, mask

    # Choose the largest contour (assuming single object in mask)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour with a Bezier curve
    approx_curve = cv2.approxPolyDP(largest_contour, 0.01 * cv2.arcLength(largest_contour, True), True)
    approx_curve = approx_curve.squeeze(1)

    # random sample points
    num_points = max(num_points,len(approx_curve))
    point_idx = np.random.choice(np.arange(len(approx_curve)),num_points,replace=False)
    approx_curve = approx_curve[point_idx]

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0] # 2500x2

    # find nearest points of the curve
    point_idx = pairwise_distances_argmin_min(approx_curve,src)[0]

    # add sinusoidal oscillation to row coordinates
    move = np.random.randint(-max_translation,max_translation,size=(num_points,2))
    dst = src.copy()
    for i in range(num_points):
        cur_idx = point_idx[i]
        dst[cur_idx] += move[i]

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_image = warp(image, tform, output_shape=(rows, cols),preserve_range=True).astype(np.uint8)
    out_mask = warp(mask, tform, output_shape=(rows, cols)).astype(np.uint8)
    diff_mask = mask - out_mask*255
    diff_mask[diff_mask<0] = 0
    


    # show
    # plt.subplot(1,3,1)
    # plt.imshow(mask)
    # plt.subplot(1,3,2)
    # plt.imshow(out_mask)
    # plt.subplot(1,3,3)
    # plt.imshow(out_image)
    # plt.show()
    out = {'image': out_image, 'mask': out_mask,'inpaint_mask':diff_mask}
    return out

def labeled_ssa(source_img,source_masks,source_labelmap,target_img,target_masks,target_labelmap,target_background,config):
    # both source and target masks are labeled
    # paste source image to target image
    target_masks = split_masks_from_one_mask(target_masks)
    # filter masks that are background
    new_target_masks = list()
    for m in target_masks:
        if intersect_ratio(m,target_labelmap) > 0.9:
            new_target_masks.append(m)
    target_masks = new_target_masks
    # take numbere of rand(config['min_aug_num'],config['max_aug_num']) masks
    aug_num = np.random.randint(config['min_aug_num'],config['max_aug_num']+1)
    assert aug_num<=np.max(target_labelmap), "aug_num should be smaller than the number of classes"
    aug_classes = np.random.choice(np.arange(1,np.max(target_labelmap)+1),aug_num,replace=False)
    result_labelmap = target_labelmap.copy()
    result_img = target_img.copy()
    for aug_class in aug_classes:
        class_mask = (target_labelmap==aug_class).astype(np.uint8)*255
        class_img = np.where(class_mask[...,None]==255,target_img,0)

        # sigma = np.random.randint(3,6)
        # alpha = np.random.randint(3,6)
        # aug = A.ElasticTransform(
        #     p=1, 
        #     alpha=200, 
        #     sigma=200 * 0.01 * sigma, 
        #     alpha_affine=200 * 0.01 * alpha,
        #     interpolation=cv2.INTER_NEAREST,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0,
        #     mask_value=0
        # )
        # augmented = aug(image=class_img,mask=class_mask)
        if np.max(class_mask) == 0:
            continue
        augmented = ssa_transform(class_img,class_mask,num_points=config['num_points'],max_translation=config['max_translation'])
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        inpaint_mask = augmented['inpaint_mask']
        # fill zero to both image and mask
        result_img[class_mask>0] = 0
        result_labelmap[class_mask>0] = 0
        # paste augmented image and mask
        result_img[aug_mask>0] = aug_image[aug_mask>0]
        # fill diff_mask with sorrounding values
        result_img = cv2.inpaint(result_img,inpaint_mask,3,cv2.INPAINT_TELEA)
        result_labelmap[aug_mask>0] = aug_class
        # plt.subplot(2,2,1)
        # plt.imshow(aug_image)
        # plt.subplot(2,2,2)
        # plt.imshow(aug_mask)
        # plt.subplot(2,2,3)
        # plt.imshow(result_img)
        # plt.subplot(2,2,4)
        # plt.imshow(result_labelmap)
        # plt.show()
        # print()
    
    ###############################
    # visualize
    # color_labelmap = np.zeros_like(result_img)
    # for i in range(1,np.max(result_labelmap)+1):
    #     color_labelmap[result_labelmap==i] = color_list[i-1][::-1]
    # vis_image = np.hstack([result_img,cv2.cvtColor(result_labelmap,cv2.COLOR_GRAY2BGR),color_labelmap])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)

    return result_img, result_labelmap#, vis_image

def lsa(source_img,source_mask,background,target_img,config):
    source_masks = split_masks_from_one_mask(source_mask)
    new_source_mask = list()
    for m in source_masks:
        if intersect_ratio(m,background) > 0.9:
            new_source_mask.append(m)
    source_masks = new_source_mask

    aug_num = 0
    final_mask = np.zeros_like(source_img[:,:,0])
    result_img = target_img.copy()
    while aug_num < config['min_aug_num']:
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        for attempt in range(500):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,background) < 0.5:
                # found a proper position
                # print(f"intersect ratio:{intersect_ratio(result_mask,target_background)}")
                break
        # print(f"attempt:{attempt}")
        if attempt >= 500:
            # failed to find a proper position
            continue
        else:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            aug_num += 1

    # visualize
    # vis_image = np.hstack([source_img,cv2.cvtColor(source_mask,cv2.COLOR_GRAY2BGR),result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR)])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)
    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
            
    for i in range(config['max_aug_num']-config['min_aug_num']):
        source_mask,idx = sample_mask(source_masks,config['weight_power'])
        source_masks.pop(idx)
        if (source_labelmap[source_mask>0]==0).sum() > 100:
            continue
        for attempt in range(500+1):
            if attempt+1 % 10 == 0:
                # re-sample source mask
                source_mask,idx = sample_mask(source_masks,config['weight_power'])
                source_masks.pop(idx)
                if (source_labelmap[source_mask>0]==0).sum() > 100:
                    continue
            bbox = cv2.boundingRect(source_mask)
            source_x = bbox[0]+bbox[2]//2
            source_y = bbox[1]+bbox[3]//2
            target_point = sample_point(source_x/target_img.shape[1],source_y/target_img.shape[0],
                                    min_distance=config['min_distance'],
                                    boundary=config['boundary'])
            target_x = target_point[0]*target_img.shape[1]
            target_y = target_point[1]*target_img.shape[0]
            delta_x = target_x-source_x
            delta_y = target_y-source_y
            rotate_matrix = cv2.getRotationMatrix2D([source_x,source_y], np.random.randint(0,360), 1.0)
            rotate_matrix = np.vstack([rotate_matrix,np.array([0,0,1])])
            translation_matrix = np.array([[1,0,delta_x],[0,1,delta_y]])
            translation_matrix = np.vstack([translation_matrix,np.array([0,0,1])])
            affine_matrix = np.dot(translation_matrix,rotate_matrix)
            affine_matrix = affine_matrix[:2]
            result_mask = cv2.warpAffine(source_mask,
                                            affine_matrix,
                                            (source_mask.shape[1],source_mask.shape[0]))
            result_mask[result_mask>200] = 255
            result_mask[result_mask<=200] = 0

            if intersect_ratio(result_mask,target_background) < 0.9:
                # found a proper position
                break

        # print(f"attempt:{attempt}")
        if attempt < 500:
            temp_img = np.zeros_like(source_img)
            temp_img[source_mask>0] = source_img[source_mask>0]
            temp_img = cv2.warpAffine(temp_img,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            temp_labelmap = np.zeros_like(source_labelmap)
            temp_labelmap[source_mask>0] = source_labelmap[source_mask>0]
            temp_labelmap = cv2.warpAffine(temp_labelmap,
                                    affine_matrix,
                                    (source_mask.shape[1],source_mask.shape[0]))
            result_labelmap[result_mask>0] = temp_labelmap[result_mask>0]
            result_img[result_mask>0] = temp_img[result_mask>0]
            final_mask[result_mask>0] = 255
            
            aug_num += 1
    ###############################
    # visualize
    # color_labelmap = np.zeros_like(result_img)
    # for i in range(1,np.max(result_labelmap)+1):
    #     color_labelmap[result_labelmap==i] = color_list[i-1][::-1]
    # vis_image = np.hstack([result_img,cv2.cvtColor(final_mask,cv2.COLOR_GRAY2BGR),color_labelmap])
    # vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)

    # cv2.imshow("result",vis_image)
    # cv2.waitKey(0)
    #cv2.imwrite("result.png",vis_image)

    return result_img, result_labelmap#, vis_image

    



class LSA():
    def __init__(self,images,masks,label_maps,backgrounds,config):
        self.images = images
        self.masks = masks
        self.label_maps = label_maps
        self.backgrounds = backgrounds
        self.config = config
    
    def augment(self,idx):
        target_img = self.images[idx]
        target_masks = self.masks[idx]
        target_background = self.backgrounds[idx]
        target_labelmap = self.label_maps[idx]

        source_idx = random.choice(np.arange(len(self.images)))
        source_img = self.images[idx]
        source_masks = self.masks[idx]
        source_labelmap = self.label_maps[idx]
        # source_background = self.backgrounds[source_idx]

        result_img, result_labelmap = labeled_lsa(source_img,
                                               source_masks,
                                               np.ones_like(source_labelmap),
                                               target_img,
                                               target_masks,
                                               np.zeros_like(target_labelmap),
                                               target_background,
                                               self.config)

        return result_img, result_labelmap



class LabeledLSA():
    def __init__(self,images,masks,label_maps,backgrounds,config):
        self.images = images
        self.masks = masks
        self.label_maps = label_maps
        self.backgrounds = backgrounds
        self.config = config
    
    def augment(self,idx):
        target_img = self.images[idx]
        target_masks = self.masks[idx]
        target_background = self.backgrounds[idx]
        target_labelmap = self.label_maps[idx]

        source_idx = idx#random.choice(np.arange(len(self.images)))
        source_img = self.images[idx]
        source_masks = self.masks[idx]
        source_labelmap = self.label_maps[idx]
        # source_background = self.backgrounds[source_idx]

        result_img, result_labelmap = labeled_lsa(source_img,
                                               source_masks,
                                               source_labelmap,
                                               target_img,
                                               target_masks,
                                               target_labelmap,
                                               target_background,
                                               self.config)

        return result_img, result_labelmap
    

class LabeledSSA():
    def __init__(self,images,masks,label_maps,backgrounds,config):
        self.images = images
        self.masks = masks
        self.label_maps = label_maps
        self.backgrounds = backgrounds
        self.config = config
    
    def augment(self,idx):
        target_img = self.images[idx]
        target_masks = self.masks[idx]
        target_background = self.backgrounds[idx]
        target_labelmap = self.label_maps[idx]

        source_idx = random.choice(np.arange(len(self.images)))
        source_img = self.images[idx]
        source_masks = self.masks[idx]
        source_labelmap = self.label_maps[idx]
        # source_background = self.backgrounds[source_idx]

        result_img, result_labelmap = labeled_ssa(source_img,
                                               source_masks,
                                               source_labelmap,
                                               target_img,
                                               target_masks,
                                               target_labelmap,
                                               target_background,
                                               self.config)

        return result_img, result_labelmap
    

if __name__ == "__main__":
    from rich.progress import track
    category = 'breakfast_box'
    data_root = "C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets"
    img_paths = glob.glob(f"{data_root}/mvtec_loco_anomaly_detection/{category}/train/good/*.png")[:30]
    mask_paths = glob.glob(f"{data_root}/masks/{category}/*/refined_masks.png")[:30]
    labelmap_paths = glob.glob(f"{data_root}/masks/{category}/*/filtered_cluster_map.png")[:30]
    background_paths = glob.glob(f"{data_root}/masks/{category}/*/background.jpg")[:30]
    
    images = [cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB) for p in track(img_paths,description="loading images...")]
    masks = [cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in track(mask_paths,description="loading masks...")]
    backgrounds = [cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in track(background_paths,description="loading backgrounds...")]
    label_maps = [cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in track(labelmap_paths,description="loading labelmaps...")]

    images = [cv2.resize(img,(256,256)) for img in images]
    masks = [cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST) for mask in masks]
    backgrounds = [cv2.resize(bg,(256,256),interpolation=cv2.INTER_NEAREST) for bg in backgrounds]
    label_maps = [cv2.resize(label_map,(256,256),interpolation=cv2.INTER_NEAREST) for label_map in label_maps]

    config = {
        "min_distance":0.5,
        "max_aug_num":2,
        "min_aug_num":1,
        'boundary':0.1,
        "num_points":2,
        "max_translation":15,
    }
    lsa_ = LabeledSSA(images,masks,label_maps,backgrounds,config)

    for i in range(20):
        result_img, result_mask, vis_img = lsa_.augment(i)
        cv2.imshow("result",vis_img)
        cv2.waitKey(0)

    # import time
    # times = list()
    # for i in range(1000):
    #     s = time.time()
    #     idx = random.choice(np.arange(len(images)))
    #     result_img, result_mask, vis_img = lsa_.augment(idx)
    #     times.append(time.time()-s)
    # print(f"average time:{np.mean(times[500:])}")





