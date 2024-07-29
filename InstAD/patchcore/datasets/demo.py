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
        image_path,
        inst_path,
        classname,
        resize=256,
        split=DatasetSplit.TEST,
        seed=0,
        data_path="",
        zero_shot=False,
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
        self.image_path = image_path
        self.inst_path = inst_path
        self.mask_path = inst_path.replace(f"{classname[0]}",f"{classname[0]}_mask")
        self.split = split
        self.classnames_to_use = classname if classname is not None else _CLASSNAMES
        self.train_val_split = 1.0
        self.data_path = data_path
        self.zero_shot = zero_shot
        self.n_shot=4
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.resize = resize

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
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []
        classname = self.classnames_to_use[0]

        if self.split == DatasetSplit.TRAIN:
            raise NotImplementedError("Training split not implemented in DEMO version.")
        else:
            insts = glob.glob(os.path.join(self.inst_path, "*.*"))
            mask = self.image_path.replace("test","ground_truth").replace(".png","_mask.png").replace(".JPG","_mask.png")
            for inst in insts:
                data_to_iterate.append((
                    classname, os.path.dirname(self.image_path).split("/")[-1], inst, mask
                ))
        

        return imgpaths_per_class, data_to_iterate