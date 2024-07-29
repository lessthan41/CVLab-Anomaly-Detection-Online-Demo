import numpy as np
import cv2
import pickle
import os
import glob
import shutil
import PIL.Image as Image
import tqdm
from sklearn.cluster import MeanShift, DBSCAN


def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return

def save_pickle(path, mask):
    with open(path, 'wb') as f:
        pickle.dump(mask, f)
    return

def load_pickle(path):
    with open(path, 'rb') as f:
        mask = pickle.load(f)
    return mask

def link_string(str_list):
    output_filename = ""
    for i in str_list:
        output_filename += i + "_"
    output_filename = output_filename[:-1]
    return output_filename

def convert_to_img_path(path):
    """convert instance path to image path"""
    dir = os.path.dirname(path)
    base = os.path.basename(path).split("_")
    if len(base) == 1:
        return path
    new_base = base[0] + base[1][-4:]
    return os.path.join(dir,new_base)

def convert_to_inst_path(path, idx):
    """convert image path to instance path"""
    return f"{path[:-4]}_{idx}{path[-4:]}"

def convert_to_pos_path(path):
    """convert image path to position path"""
    return convert_to_img_path(path.replace("instance","position"))[:-4]+".pkl"

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

def gaussian_blur(mask, ksize, sigma):
    blur = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
    return blur    

def canny(mask, low=50, high=150):
    blur = cv2.medianBlur(mask, 5) 
    edges = cv2.Canny(blur,low,high)
    return edges

def erode(mask, ksize):
    kernel = np.ones((ksize,ksize),np.uint8)
    erosion = cv2.erode(mask,kernel)
    return erosion

def dilate(mask, ksize):
    kernel = np.ones((ksize,ksize),np.uint8)
    dilation = cv2.dilate(mask,kernel)
    return dilation

def fill_small_instances(mask):
    _channel = mask.ndim
    if _channel == 3:
        mask = mask[:,:,0]
    ret_mask = mask

    ### contours analysis
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ret_mask, contours, -1, 255, thickness=cv2.FILLED)

    if _channel == 3:
        ret_mask = np.stack([ret_mask,ret_mask,ret_mask],axis=2)
    return ret_mask

def remove_noise(mask):
    _channel = mask.ndim
    if _channel == 3:
        mask = mask[:,:,0]
    ret_mask = mask
    
    ### connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ret_mask, connectivity=8)
    total_size = stats[1:, -1].sum()
    threshold_size = 0.01 * total_size
    for i, size in enumerate(stats[1:, -1], start=1):
        if size < threshold_size:
            ret_mask[labels==i] = 0
    
    if _channel == 3:
        ret_mask = np.stack([ret_mask,ret_mask,ret_mask],axis=2)
    return ret_mask

def closing(mask, ksize=5):
    kernel = np.ones((ksize,ksize),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def opening(mask, ksize=5):
    kernel = np.ones((ksize,ksize),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening

def fill_border(mask):
    """set border of the mask as zero"""
    if mask.ndim == 3:
        output = mask[:,:,0]
    else:
        output = mask
    h,w = output.shape
    h_crop = int(h*0.01)
    w_crop = int(w*0.01)
    output[:h_crop,:] = 0
    output[(h-h_crop):,:] = 0
    output[:,:w_crop] = 0
    output[:,(w-w_crop):] = 0
    if mask.ndim == 3:
        output = np.stack([output,output,output],axis=2)
    return output

def overlap_ratio(mask1, mask2):
    """calculate the overlap ratio between two masks"""
    _mask1 = mask1[...,0].astype(bool) if mask1.ndim == 3 else mask1.astype(bool)
    _mask2 = mask2[...,0].astype(bool) if mask2.ndim == 3 else mask2.astype(bool)
    total_area = np.sum(_mask1)
    overlap_area = np.sum(np.logical_and(_mask1, _mask2))
    return overlap_area / total_area

def color_masks(masks):
    # if type(masks) != list:
    #     masks = [masks]
    if type(masks) == list and len(masks) == 1:
        return np.where(masks[0],255,0).astype(np.uint8)
    if type(masks) != list and len(masks.shape) == 2:
        return np.where(masks!=0,255,0).astype(np.uint8)
    color_mask = np.zeros([masks[0].shape[0],masks[0].shape[1],3],dtype=np.uint8)
    masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        color_mask[mask!=0] = np.random.randint(0,255,[3])
    return color_mask

def generate_sliding_window(img_path, window_scale, output_path="/home/tokichan/research/segment/output"):
    """generate sliding windows with window size=imgsize*window_scale for instance"""
    output_path = os.path.join(output_path, "windows")
    reset_folder(output_path)
    img = cv2.imread(img_path)
    img_id = os.path.basename(img_path).split("_")[0]
    h, w, _ = img.shape
    window_size = int(h*window_scale)
    id=0
    for i in range(0,h-window_size,window_size//10):
        for j in range(0,w-window_size,window_size//10):
            window = img[i:i+window_size, j:j+window_size]
            filename = os.path.join(output_path, f"{img_id}_{id}.png")
            cv2.imwrite(filename, window)
            id += 1
        
        ### last window (each row)
        window = img[i:i+window_size, w-window_size:w]
        filename = os.path.join(output_path, f"{img_id}_{id}.png")
        cv2.imwrite(filename, window)
        id += 1
    
    ### last window (each img)
    window = img[h-window_size:h, w-window_size:w]
    filename = os.path.join(output_path, f"{img_id}_{id}.png")
    cv2.imwrite(filename, window)

    return

def get_border(img_shape, center, size):
    h,w,c = img_shape
    cx,cy = center
    x1 = cx-size//2 if cx-size//2 >= 0 else 0
    x2 = cx+(size-size//2) if cx+(size-size//2) <= w else w
    y1 = cy-size//2 if cy-size//2 >= 0 else 0
    y2 = cy+(size-size//2) if cy+(size-size//2) <= h else h
    return y1,y2,x1,x2

def make_padding(crop_img, orig_image, borders, output_size):
    y1,y2,x1,x2 = borders
    h,w,c = orig_image.shape
    ch,cw,cc = crop_img.shape
    if h==output_size and w==output_size:
        return crop_img
    else:
        if x1==0:
            crop_img = cv2.copyMakeBorder(crop_img,0,0,output_size-cw,0,cv2.BORDER_CONSTANT,value=(0))
        elif x2==w:
            crop_img = cv2.copyMakeBorder(crop_img,0,0,0,output_size-cw,cv2.BORDER_CONSTANT,value=(0))
        if y1==0:
            crop_img = cv2.copyMakeBorder(crop_img,output_size-ch,0,0,0,cv2.BORDER_CONSTANT,value=(0))
        elif y2==h:
            crop_img = cv2.copyMakeBorder(crop_img,0,output_size-ch,0,0,cv2.BORDER_CONSTANT,value=(0))
    return crop_img

def crop_by_mask(img,mask):
    """
        crop image by mask
        img: np.array
        mask: np.array
        output: np.array
    """
    result = np.zeros_like(img,dtype=np.uint8)
    if mask.shape[-1]==3:
        mask_one_channel = mask[:,:,0]
    else:
        mask_one_channel = mask
    if result.shape[-1]==1:
        result[mask_one_channel==255] = img[mask_one_channel==255]
    elif result.shape[-1]==3:
        for i in range(3):
            result[mask_one_channel==255,i] = img[mask_one_channel==255,i]
    return result

def crop_instance(orig_img, inst_pos, target_size):
    """
        orig_img: np.array
        inst_pos: dict { center: (center_x, center_y), scale: scale }
        target_size: int
    """
    h,w,c = orig_img.shape
    cx,cy = inst_pos["center"][0], inst_pos["center"][1]
    scale = inst_pos["scale"]
    coor_cx, coor_cy = int(w*cx), int(h*cy)
    orig_size = int(target_size/scale)
    y1,y2,x1,x2 = get_border(orig_img.shape, (coor_cx,coor_cy), orig_size)
    crop_img = orig_img[y1:y2,x1:x2]
    crop_img = make_padding(crop_img, orig_img, (y1,y2,x1,x2), orig_size)
    # cv2.imwrite(f"/home/tokichan/research/segment/output/{class_name}_cropped.png", crop_img)
    return crop_img, (y1,y2,x1,x2)

def fit_mask(inst_mask, orig_img_shape, borders):
    """
        cut mask to fit the shape
        inst_mask: np.array
        orig_img_shape: tuple (h,w)
        borders: tuple (y1,y2,x1,x2)
    """
    y1,y2,x1,x2 = borders
    th,tw = abs(y2-y1) ,abs(x2-x1)
    h,w = orig_img_shape
    ih,iw = inst_mask.shape
    if th < ih:
        if y2 == h:
            inst_mask = inst_mask[:th,:]
        else: # y1 == 0
            inst_mask = inst_mask[(ih-th):,:]
    if tw < iw:
        if x2 == w:
            inst_mask = inst_mask[:,:tw]
        else:
            inst_mask = inst_mask[:,(iw-tw):]
    return inst_mask

def remove_duplicate_mask(sam_masks, gsam_mask): #, path="/home/tokichan/research/segment/output/test_color_mask.png"):
    """remove similar mask in sam_masks with gsam_mask"""
    ### discard the similar mask in masks with gsam_mask
    if gsam_mask.ndim == 3:
        gsam_mask = gsam_mask[:,:,0]
    gsam_mask = gsam_mask.astype(bool)
    gsam_area = np.sum(gsam_mask)
    ratio_list = []
    for i, mask in enumerate(sam_masks):
        mask = mask.astype(bool)
        ratio_list.append(np.sum(np.logical_and(gsam_mask, mask)))
    ratio_list = [i/gsam_area for i in ratio_list]
    del_idx = [i for i, value in enumerate(ratio_list) if value>0.9]
    sam_masks = np.delete(sam_masks, del_idx, axis=0)

    ### sort masks using mask size
    mask_size = [np.sum(mask) for mask in sam_masks]
    sam_masks = sam_masks[np.argsort(mask_size)]
    sam_masks = sam_masks[::-1] # reverse
    
    ### save color_mask
    color_mask = color_masks(sam_masks)
    # cv2.imwrite(path, color_mask)
    return sam_masks

def load_pos(pos_dir, img_id):
    """load position information for all instances in img_id"""
    pos = {}
    for i, img in enumerate(img_id):
        pos_path = os.path.join(pos_dir, img+".pkl")
        _ = load_pickle(pos_path)
        pos = pos | _
    return pos

def get_mask_area(mask):
    if mask.ndim == 3:
        mask = mask[:,:,0]
    mask = mask.astype(bool)
    return np.sum(mask)

def get_instances_number(img_id, instance_dirs):
    """get len(instances) in img_id from instance_dir"""
    instances = []
    for instance_dir in instance_dirs:
        instances += glob.glob(os.path.join(instance_dir, "*.*"))
    _instances = [i for i in instances if os.path.basename(i).split("_")[0] in img_id]
    return len(_instances)

def mean_shift(areas, b=None):
    meanshift = MeanShift() if b is None else MeanShift(bandwidth=b)
    cluster_labels = meanshift.fit_predict(areas.reshape(-1,1))
    cluster_centers = meanshift.cluster_centers_
    return cluster_labels, cluster_centers

def dbscan(mean_feats, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(mean_feats)
    return cluster_labels

def detect_outliers(areas, b=None):
    """b: bandwidth of sigma"""
    areas = np.array(areas)
    mean = np.mean(areas)
    std_dev = np.std(areas)
    z_scores = (areas - mean) / std_dev
    cluster_labels = np.zeros_like(areas)
    cluster_labels[(z_scores > (3.0-b)) | (z_scores < (-3.0+b))] = 1
    return cluster_labels, []
    

def background_fill(class_name, instance_path, mask_path, mode="train"):
    """ fill background with mean color """
    
    ### compute mean background color using the first image in training set
    # construct reference color
    _path = os.path.join(instance_path,"train",class_name,"good")
    _ref_dir = os.listdir(_path)
    _ref_dir.sort()
    ref_path = os.path.join(_path,_ref_dir[0])
    ref_mask_path = ref_path.replace(instance_path, mask_path)
    ref_instance = Image.open(ref_path).convert("RGB")
    ref_instance = np.array(ref_instance)
    ref_mask = Image.open(ref_mask_path).convert("RGB")
    ref_mask = np.array(ref_mask)

    ref_instance = ref_instance.reshape(-1,3)
    ref_instance = ref_instance[~np.all(ref_instance==0,axis=1)]
    mean_color = np.mean(ref_instance,axis=0).astype(np.uint8)

    ### fill background using mean background color in "mode" set
    defect_types = os.listdir(os.path.join(instance_path,mode,class_name))
    for defect_type in defect_types:
        print(f"'{defect_type}' Instance Background Filling...")
        instances = os.listdir(os.path.join(instance_path,mode,class_name,defect_type))
        instances.sort()
        for i in tqdm.tqdm(instances):
            i_path = os.path.join(instance_path,mode,class_name,defect_type,i)
            i_mask_path = i_path.replace(instance_path, mask_path)
            instance = Image.open(i_path).convert("RGB")
            instance = np.array(instance)
            mask = Image.open(i_mask_path).convert("RGB")
            mask = np.array(mask)
            instance[np.all(mask!=255,axis=2)] = mean_color
            instance = Image.fromarray(instance)
            instance.save(i_path)

    return