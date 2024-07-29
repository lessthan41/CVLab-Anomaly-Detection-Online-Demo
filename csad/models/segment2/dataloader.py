from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import tqdm
from torchvision import transforms
import glob
from scipy.ndimage import binary_fill_holes
from rich.progress import track
from models.segment2.lsa import LabeledLSA, LabeledSSA
import albumentations as A

def merge_masks(masks):
    # remove empty masks
    new_mask = list()
    for i,mask in enumerate(masks):
        if np.sum(mask) > 0:
            new_mask.append(mask)
    masks = new_mask


    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    sorted_masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    mask_sum = np.array([np.sum(mask) for mask in masks])
    mask_order = np.argsort(mask_sum)[::-1]
    mask_map = {order+1:i+1 for i,order in enumerate(mask_order)}
    mask_map[0] = 0
    for i,mask in enumerate(sorted_masks):
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(i+1)

    new_mask = np.zeros_like(result_mask)
    for i, order in enumerate(mask_order+1):
        new_mask[result_mask==order] = mask_map[order]
    return new_mask

def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        #print(np.sum(mask>0))
        if np.sum(mask!=0) > 100:
            result_masks.append(mask)
    return result_masks
class Padding2Resize():
    def __init__(self, pad_l, pad_t, pad_r, pad_b):
        self.pad_l = pad_l
        self.pad_t = pad_t
        self.pad_r = pad_r
        self.pad_b = pad_b

    def __call__(self,image,target_size,mode='nearest'):
        shape = len(image.shape)
        if shape == 3:
            image = image[None,:,:,:]
        elif shape == 2:
            image = image[None,None,:,:]
        # B,C,H,W
        if self.pad_b == 0:
            image = image[:,:,self.pad_t:]
        else:
            image = image[:,:,self.pad_t:-self.pad_b]
        if self.pad_r == 0:
            image = image[:,:,:,self.pad_l:]
        else:
            image = image[:,:,:,self.pad_l:-self.pad_r]
        
        if isinstance(image,np.ndarray):
            image = cv2.resize(image,(target_size,target_size),interpolation=cv2.INTER_NEAREST if mode == 'nearest' else cv2.INTER_LINEAR)
        elif isinstance(image,torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(target_size,target_size), mode=mode)


        if shape == 3:
            return image[0]
        elif shape == 2:
            return image[0,0]
        return image
    
def get_padding_functions(orig_size,target_size=256,resize_target_size=None,mode='nearest',fill=0):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
    resize_target_size = target_size if resize_target_size is None else resize_target_size
    imsize = orig_size
    long_size = max(imsize)
    scale = target_size / long_size
    new_h = int(imsize[1] * scale + 0.5)
    new_w = int(imsize[0] * scale + 0.5)

    if (target_size - new_w) % 2 == 0:
        pad_l = pad_r = (target_size - new_w) // 2
    else:
        pad_l,pad_r = (target_size - new_w) // 2,(target_size - new_w) // 2 + 1
    if (target_size - new_h) % 2 == 0:
        pad_t = pad_b = (target_size - new_h) // 2
    else:
        pad_t,pad_b = (target_size - new_h) // 2,(target_size - new_h) // 2 + 1
    inter =  transforms.InterpolationMode.NEAREST if mode == 'nearest' else transforms.InterpolationMode.BILINEAR

    padding_func = transforms.Compose([
        transforms.Resize((new_h,new_w),interpolation=inter),
        transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=fill, padding_mode='constant')
    ])
    return padding_func, Padding2Resize(pad_l,pad_t,pad_r,pad_b)

class SegmentDataset(Dataset):
    def __init__(self,image_path,mask_root,sup=True,image_size=256,config=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.sup = sup
        # print(f"Building {'supervise' if sup else 'unsupervise'} dataset...")
        self.image_paths = glob.glob(image_path)
        self.info_path = mask_root+"/info/"
        self.gt_paths = glob.glob(mask_root+"/*/filtered_cluster_map.png")
        self.segment_paths = glob.glob(mask_root+"/*/refined_masks.png")
        self.background_paths = glob.glob(mask_root+"/*/background.jpg")
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        


        self.image_dataset = self.image_paths
        self.padding_func, self.pad2resize = get_padding_functions(Image.open(self.image_dataset[0]).size,target_size=image_size)
        # self.padding_func_linear, _ = get_padding_functions(Image.open(self.image_dataset[0]).size,target_size=image_size,mode='bilinear')
        self.background_padding_func, self.background_inverse_padding_func = get_padding_functions(Image.open(self.background_paths[0]).size,target_size=image_size,fill=255)
        
        self.good_indices = np.sort(np.load(self.info_path+"filtered_histogram_indices.npy")).tolist()
        self.is_good_indices = np.array([1 if i in self.good_indices else 0 for i in range(len(self.image_paths))])
        
        

        if self.sup:
            self.image_paths = [image_path for i,image_path in enumerate(self.image_paths) if i in self.good_indices]
            self.gt_paths = [gt_path for i,gt_path in enumerate(self.gt_paths) if i in self.good_indices]
        else:
            self.image_paths = [image_path for i,image_path in enumerate(self.image_paths) if i not in self.good_indices]
            self.gt_paths = [gt_path for i,gt_path in enumerate(self.gt_paths) if i not in self.good_indices]
        
        
        self.image_dataset = []
        self.background_dataset = []
        self.gt_dataset = []
        self.transform = self.create_transform()
        
        for i, image_path in tqdm.tqdm(list(enumerate(self.image_paths)),desc=f'loading {"sup" if sup else "unsup"} datas...'):
            image = self.padding_func(Image.open(image_path).convert("RGB"))
            if self.config['fill_holes']:
                gt = np.array(Image.open(self.gt_paths[i]).convert("L"))
                gt = split_masks_from_one_mask(gt)
                gt = merge_masks([binary_fill_holes(mask) if self.config['fill_holes'] else mask for mask in gt])
                gt = self.padding_func(Image.fromarray(gt))
            else:
                gt = self.padding_func(Image.open(self.gt_paths[i]).convert("L"))
            self.image_dataset.append(np.array(image))
            self.gt_dataset.append(np.array(gt))
            
        ########################
        # prepare for LSA
        if self.sup:
            # self.lsa_images = [np.array(padding_func(Image.open(image_path).convert("RGB"))) for i,image_path in enumerate(self.image_dataset) if i in self.good_indices]
            self.lsa_images = self.image_dataset
            self.lsa_backgrounds = [np.array(self.background_padding_func(Image.open(image_path).convert("L"))) for i,image_path in enumerate(self.background_paths) if i in self.good_indices]
            # self.lsa_labels = [np.array(padding_func(Image.open(image_path).convert("L"))) for i,image_path in enumerate(self.gt_paths) if i in self.good_indices]
            self.lsa_labels =  self.gt_dataset
            self.lsa_masks = [np.array(self.padding_func(Image.open(image_path).convert("L"))) for i,image_path in  enumerate(self.segment_paths) if i in self.good_indices]
            self.lsa_ratio = self.config['LSA_ratio']
            self.ssa_ratio = self.config['SSA_ratio']
            self.lsa = LabeledLSA(
                self.lsa_images,
                self.lsa_masks,
                self.lsa_labels,
                self.lsa_backgrounds,
                self.config['LSA_config']
            )
            self.ssa = LabeledSSA(
                self.lsa_images,
                self.lsa_masks,
                self.lsa_labels,
                self.lsa_backgrounds,
                self.config['LSA_config']
            )
        ########################
        
        

    def create_transform(self):
        return A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=0.01,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0
            ),
            # A.RGBShift(
            #     r_shift_limit=5,
            #     g_shift_limit=5,
            #     b_shift_limit=5,
            #     p=1.0
            # ),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.05,
                p=1.0
            ),
            # A.GridDistortion(
            #     num_steps=20,
            #     distort_limit=0.1,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=0,
            #     mask_value=0,
            #     p=1.0
            # ),
        ])
    
    def apply_transform(self,image,mask,transform):
        transformed = transform(image=image,mask=mask)
        return transformed['image'],transformed['mask']



    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        image = self.image_dataset[index]
        gt = self.gt_dataset[index]
        gt_path = self.gt_paths[index]
        rand_gt = self.gt_dataset[np.random.randint(len(self.gt_dataset))]


        if self.sup:
            # only augment in supervised mode
            if np.random.rand() > 1.0 - self.lsa_ratio:
                image, gt = self.lsa.augment(index)
            if np.random.rand() > 1.0 - self.ssa_ratio:
                image, gt, vis_image = self.ssa.augment(index)
            image, gt = self.apply_transform(image,gt,self.transform)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)

        image = self.norm_transform(image)
        gt = torch.unsqueeze(torch.from_numpy(gt),dim=0).type(torch.long)
        rand_gt = torch.unsqueeze(torch.from_numpy(rand_gt),dim=0).type(torch.long)

        image = image.to(self.device)
        gt = gt.to(self.device)
        rand_gt = rand_gt.to(self.device)

        return image, gt, rand_gt, gt_path
    
    
class SegmentDatasetTest(Dataset):
    def __init__(self,image_path,image_size=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            #transforms.Resize((image_size,image_size),interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = glob.glob(image_path)

        padding_func, inverse_padding_func = get_padding_functions(Image.open(self.image_paths[0]).size,target_size=image_size)
        self.image_dataset = [transform(padding_func(Image.open(image_path).convert("RGB"))).to(self.device) for image_path in tqdm.tqdm(self.image_paths,desc='loading images...')]
        

    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        image = self.image_dataset[index]
        path = self.image_paths[index]
        return image,path
    
def de_normalize(tensor):
    # tensor: (B,C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image = Image.open("C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/juice_bottle/train/good/000.png")

    image = transforms.ToTensor()(image)
    resize_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(256,256), mode='bilinear')
    padding_func, inverse_padding_func = get_padding_functions(image.shape[-2:],target_size=256)
    image2 = padding_func(image)
    image3 = inverse_padding_func(image2.unsqueeze(0))
    image3 = image3[0]

    resize_image = resize_image[0].permute(1,2,0).cpu().numpy()
    image3 = image3.permute(1,2,0).cpu().numpy()
    plt.subplot(1,2,1)
    plt.imshow(resize_image)
    plt.subplot(1,2,2)

    plt.imshow(image3)
    plt.show()
    print(resize_image == image3)
    print()














    category = 'breakfast_box'
    config = {
        "train_image_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{category}/train/good/*.png",
        "test_image_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{category}/test/*/*.png",
        "mask_root":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/masks/{category}",
        "model_path":"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/ckpt/segmentor.pth ",
        "fill_holes":True,
        "in_dim":[256,1024],
        "load":False,
        "image_size":256,
        "lr":1e-4,
        "epoch":150,
        "sup_only_epoch":10,
        "loss_weight":{
            "ce":1,
            "focal":1,
            "dice":1,
            "hist":1,
        },
        "LSA_config":{
            "min_distance":0.5,
            "max_aug_num":3,
            "min_aug_num":1,
            'boundary':0.1,
            "weight_power":0.0,
        }
        
    }
    sup_dataset = SegmentDataset(
        image_path=config['train_image_path'],
        sup=True,
        mask_root=config['mask_root'],
        image_size=config['image_size'],
        config=config
    )
    dataloader = DataLoader(sup_dataset,batch_size=1,shuffle=True)
    color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
    for image, gt, rand_gt, gt_path in dataloader:
        image = de_normalize(image)
        image = image[0].cpu().numpy().transpose(1,2,0)
        image = (image*255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gt = gt[0,0].cpu().numpy()
        mask = np.zeros((gt.shape[0],gt.shape[1],3),dtype=np.uint8)
        for i in range(1,np.max(gt)+1):
            mask[gt==i] = color_list[i-1]
        t = gt_path[0].split('\\')[-2]
        cv2.imshow(f"{t}",np.hstack([image,mask]))
        cv2.waitKey(0)
        print(image.shape)
        print(gt.shape)