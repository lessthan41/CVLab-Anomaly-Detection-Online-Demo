import os
from enum import Enum
import glob
import PIL
import torch
import random
from torchvision import transforms

_CLASSNAMES = [
    "candle",
    "capsules",
    "macaroni1",
    "macaroni2",
    "tubes",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class VisADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for VisA.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        augment=False,
        data_path=None,
        zero_shot=False,
        n_shot=4,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.data_path = data_path
        self.augment = augment
        self.zero_shot = zero_shot
        self.n_shot=n_shot
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.resize = resize
        self.aug_per_img = 10 if self.augment else 1


        if self.split == DatasetSplit.TRAIN and self.augment:
            self.transform_img = [
                transforms.Resize([resize,resize]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180), ### 0330 Rotation Augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        else:
            self.transform_img = [
                transforms.Resize([resize,resize]),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize([self.resize,self.resize]),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, self.resize, self.resize)

    def __getitem__(self, idx):
        idx = idx % len(self.data_to_iterate)
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        input_shape = image.size # (width, height)
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        if self.split == DatasetSplit.TRAIN:
            return len(self.data_to_iterate) * self.aug_per_img
        else:
            return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        data_to_iterate = []

        ### zero shot
        if self.zero_shot:
            if self.split == DatasetSplit.TRAIN: ### train
                for classname in self.classnames_to_use:
                    print("ZERO SHOT")
                    print("classname:", classname, "  mode:", self.split, "  instances:", len(data_to_iterate))
                return imgpaths_per_class, data_to_iterate
            elif self.split == DatasetSplit.TEST: ### test
                for classname in self.classnames_to_use:
                    inst_classpath = os.path.join(self.source, classname, self.split.value)
                    img_classpath = os.path.join(self.data_path, classname, self.split.value)
                    anomaly_types = os.listdir(inst_classpath)

                    imgpaths_per_class[classname] = {}
                    maskpaths_per_class[classname] = {}

                    for anomaly in anomaly_types:
                        imgpaths_per_class[classname][anomaly] = []
                        imgs = os.listdir(os.path.join(img_classpath, anomaly))
                        imgs = [os.path.join(img_classpath, anomaly, x) for x in imgs]
                        imgs = sorted(imgs)
                        for img in imgs:
                            path = os.path.join(inst_classpath, anomaly, os.path.basename(img))
                            _insts = glob.glob(os.path.splitext(path)[0] + "*.*")
                            _insts = sorted(_insts)
                            fname, ext = os.path.splitext(img)
                            mask = None if anomaly=="good" else fname.replace("test","ground_truth")+"_mask"+ext
                            imgpaths_per_class[classname][anomaly].append(_insts)
                            data_to_iterate.append([
                                classname, anomaly, _insts, mask
                            ])

                return imgpaths_per_class, data_to_iterate

        ### few shot
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.data_path, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                if self.split == DatasetSplit.TRAIN:
                    img_ids = [x.split("_")[0] for x in anomaly_files]
                    img_ids = list(set(img_ids))
                    img_ids = random.sample(img_ids, self.n_shot)
                    anomaly_files = [x for x in anomaly_files if x.split("_")[0] in img_ids]
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        postfix = maskpaths_per_class[classname][anomaly][0][-4:]
                        img_id = os.path.basename(image_path).split("_")[0]+"_mask"
                        maskpath = os.path.join(self.data_path,classname,"ground_truth",anomaly,img_id+postfix)
                        data_tuple.append(maskpath)
                        # data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        ### prune train instances numbers
        if self.split == DatasetSplit.TRAIN:
            if len(data_to_iterate) > 400:
                data_to_iterate = random.sample(data_to_iterate, 400)
        
        print("classname:", classname, "  mode:", self.split, "  instances:", len(data_to_iterate))

        return imgpaths_per_class, data_to_iterate


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, classname, instance_path_list, resize=256, augment=False):
        super().__init__()
        self.split = DatasetSplit.TRAIN
        self.augment = augment
        self.resize = resize
        self.data_to_iterate = [[classname, "good", i, None] for i in instance_path_list]
        self.aug_per_img = 10 if self.augment else 1
        if self.split == DatasetSplit.TRAIN and self.augment:
            self.transform_img = [
                transforms.Resize([resize,resize]),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(180), ### 0330 Rotation Augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        else:
            self.transform_img = [
                transforms.Resize([resize,resize]),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        self.transform_img = transforms.Compose(self.transform_img)
        return
    
    def __getitem__(self, idx):
        idx = idx % len(self.data_to_iterate)
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        mask = torch.zeros([1, *image.size()[1:]])
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate) * self.aug_per_img