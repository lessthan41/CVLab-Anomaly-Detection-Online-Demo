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
from dataset import augs_TIBA as img_trsform
import yaml
def merge_masks(masks):
    # remove empty masks
    new_mask = list()
    for i,mask in enumerate(masks):
        if np.sum(mask) > 0:
            new_mask.append(mask)
    masks = new_mask

    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(i+1)
    return result_mask

def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        #print(np.sum(mask>0))
        if np.sum(mask!=0) > 100:
            result_masks.append(mask)
    return result_masks

def get_padding_functions(orig_size,target_size=256,mode='nearest',fill=0):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
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
    inverse_padding_func = transforms.Compose([
        transforms.CenterCrop((new_h,new_w)),
        transforms.Resize((target_size,target_size),interpolation=inter)
    ])
    return padding_func, inverse_padding_func

class SegmentDatasetVal(Dataset):
    def __init__(self,dataset_root):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_paths = glob.glob(dataset_root+"/validation/good/*.png")
        # transform = transforms.Compose([
        #     #transforms.Resize((image_size,image_size),interpolation=Image.BILINEAR),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # padding_func, inverse_padding_func = get_padding_functions(Image.open(self.image_paths[0]).size,target_size=image_size)
        to_tensor = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        self.image_dataset = [to_tensor(Image.open(image_path).convert("RGB")).to(self.device) for image_path in track(self.image_paths,description='loading images...')]
        

    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index):
        return self.image_dataset[index]
    
class SegmentDatasetTrain(Dataset):
    def __init__(self, image_paths,label_paths,image_size, trs_form, trs_form_strong=None):
        super(SegmentDatasetTrain, self).__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform_weak = trs_form
        self.transform_strong = trs_form_strong
        self.image_size = image_size
        self.trf_normalize = self._get_to_tensor_and_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        self.padding_func, self.inverse_padding_func = get_padding_functions(Image.open(self.image_paths[0]).size,target_size=image_size)
        # self.images = [self.padding_func(Image.open(image_path).convert("RGB")) for image_path in track(self.image_paths,description='loading images...')]
        # self.labels = [self.padding_func(Image.open(label_path).convert("L")) for label_path in track(self.label_paths,description='loading labels...')]
        self.images = [Image.open(image_path).convert("RGB") for image_path in track(self.image_paths,description='loading images...')]
        self.labels = [Image.open(label_path).convert("L") for label_path in track(self.label_paths,description='loading labels...')]
        
        # random.seed(seed)

    @staticmethod
    def _get_to_tensor_and_normalize(mean, std):
        return img_trsform.ToTensorAndNormalize(mean, std)

    def __getitem__(self, index):
        # load image and its label
        image = self.images[index]
        label = self.labels[index]
        label = np.array(label)
        label = split_masks_from_one_mask(label)
        label = merge_masks([binary_fill_holes(mask) for mask in label])
        label = Image.fromarray(label)

        if self.transform_strong is None:
            image, label = self.transform_weak(image, label)
            # print(image.shape, label.shape)
            image, label = self.trf_normalize(image, label)
            return index, image, image.clone(), label
        else:
            # apply augmentation
            image_weak, label = self.transform_weak(image, label)
            image_strong = self.transform_strong(image_weak)
            # print("="*100)
            # print(index, image_weak.size, image_strong.size, label.size)
            # print("="*100)

            image_weak, label = self.trf_normalize(image_weak, label)
            image_strong, _ = self.trf_normalize(image_strong, label)
            # print(index, image_weak.shape, image_strong.shape,label.shape)

            return index, image_weak, image_strong, label

    def __len__(self):
        return len(self.image_paths)
    
def build_basic_transfrom(cfg, split="val", mean=[0.485, 0.456, 0.406]):
    ignore_label = cfg["ignore_label"]
    trs_form = []
    if split != "val":
        if cfg.get("rand_resize", False):
            trs_form.append(img_trsform.Resize(cfg.get("resize_base_size", [1024, 2048]), cfg["rand_resize"]))
        
        if cfg.get("flip", False):
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))
    
        # crop also sometime for validating
        if cfg.get("crop", False):
            crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
            trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=0))

    return img_trsform.Compose(trs_form)

def build_additional_strong_transform(cfg):
    assert cfg.get("strong_aug", False) != False
    strong_aug_nums = cfg["strong_aug"].get("num_augs", 2)
    flag_use_rand_num = cfg["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug


def get_semi_loaders(config,image_path,mask_root,image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = glob.glob(image_path)
    info_path = mask_root+"/info/"
    label_paths = glob.glob(mask_root+"/*/filtered_cluster_map.png")
    
    good_indices = np.sort(np.load(info_path+"filtered_histogram_indices.npy")).tolist()
    is_good_indices = np.array([1 if i in good_indices else 0 for i in range(len(label_paths))])


    trs_form = build_basic_transfrom(config, split="train")
    trs_form_strong = build_additional_strong_transform(config)


    good_image_paths = [image_paths[i] for i in range(len(image_paths)) if is_good_indices[i] == 1]
    good_label_paths = [label_paths[i] for i in range(len(label_paths)) if is_good_indices[i] == 1]
    bad_image_paths = [image_paths[i] for i in range(len(image_paths)) if is_good_indices[i] == 0]
    bad_label_paths = [label_paths[i] for i in range(len(label_paths)) if is_good_indices[i] == 0]

    sup_dataset = SegmentDatasetTrain(image_paths=good_image_paths,
                                        label_paths=good_label_paths,
                                        image_size=image_size,
                                        trs_form=trs_form,
                                        trs_form_strong=None)

    unsup_dataset = SegmentDatasetTrain(image_paths=bad_image_paths,
                                        label_paths=bad_label_paths,
                                        image_size=image_size,
                                        trs_form=trs_form,
                                        trs_form_strong=trs_form_strong)
    
    val_dataset = SegmentDatasetVal(
        dataset_root=image_path[:-17],
    )
    
    # val_dataset = SegmentDatasetVal(
    #     dataset_root=,
    #     image_size=512
    # )
    print(len(sup_dataset))
    print(len(unsup_dataset))

    sup_loader = DataLoader(
        sup_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    unsup_loader = DataLoader(
        unsup_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )


    return sup_loader, unsup_loader,val_loader
def de_normalize(tensor):
        device = tensor.device
        # tensor: (B,C,H,W)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return tensor

def visualize_image(image):
    # C,H,W
    # denormalize

    image = de_normalize(image)[0]
    image = image.permute(1,2,0).cpu().numpy()
    image = (image*255).astype(np.uint8)
    Image.fromarray(image).show()

def visualize_label(label):
    color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
    label = label.cpu().numpy()
    color_label = np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)
    for i in range(np.max(label)+1):
        color_label[label==i] = color_list[i]
    color_label = color_label.astype(np.uint8)
    Image.fromarray(color_label).show()
    
if __name__ == "__main__":
    category = "breakfast_box"
    image_path = f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{category}/train/good/*.png"
    mask_root = f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/masks/{category}"
    config_path = "C:/Users/kev30/Desktop/anomaly/EfficientAD-res/models/segment3/config_semi.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    sup_loader, unsup_loader, val_loader = get_semi_loaders(config['dataset'],image_path,mask_root,image_size=1024)
    for index, image, image2, label in sup_loader:
        print(image.shape)
        print(image2.shape)
        print(label.shape)
        visualize_image(image[0])
        visualize_image(image2[0])
        visualize_label(label[0])

    for index, image, image2, label in unsup_loader:
        print(image.shape)
        print(image2.shape)
        print(label.shape)
        break
