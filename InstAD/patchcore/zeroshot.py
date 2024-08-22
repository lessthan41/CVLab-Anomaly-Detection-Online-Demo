import os
import PIL
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN

from .backbones import load
from .sampler import ApproximateGreedyCoresetSampler
from .common import NearestNeighbourScorer, FaissNN, NetworkFeatureAggregator

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_backbone(backbone="resnet18", layers_to_extract_from=["layers2","layers3"], imagesize=256, device="cuda"):
    backbone = load(backbone)
    backbone.to(device)
    backbone.eval()
    feature_aggregator = NetworkFeatureAggregator(
        backbone, layers_to_extract_from, device=device, imagesize=imagesize
    )
    _ = feature_aggregator.eval()
    return feature_aggregator

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagesize=256,
        instances=[],
        split=None,
        data_path=None,
        class_name=None,
        **kwargs,
    ):
        super().__init__()
        self.data_to_iterate = instances
        self.transform_img = [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(180),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Resize([imagesize,imagesize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize([imagesize,imagesize]),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
        self.split = split
        self.data_path = data_path
        self.classnames_to_use = [class_name]
        self.augment_num = 1
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        image_path = self.data_to_iterate[idx % len(self.data_to_iterate)]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        if self.split == "test" and "good" not in image_path:
            try:
                self.mask_path = os.path.join(
                    self.data_path, 
                    image_path.split("/instance/")[1].replace("test","ground_truth"),
                )
                _ = os.path.basename(mask_path)
                # mask_path = os.path.join(os.path.dirname(mask_path), _.split("_")[0] + "_mask" + _[-4:])
                mask_path = os.path.join(os.path.dirname(mask_path), _.split("_")[0] + "_mask.png")
                mask = PIL.Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)
            except:
                mask = torch.zeros([1, *image.size()[1:]])
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": 1 if "good" not in image_path else 0,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate) * self.augment_num

class FeatureCluster():
    def __init__(self, backbone="YuShuanPatch", patchcore=None):
        self.p = 0.1
        self.sampler = ApproximateGreedyCoresetSampler(self.p, "cuda", log=False)
        # self.sampler = IdentitySampler()
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=3, nn_method=FaissNN(False, 4),
        )
        self.batch_size = 4
        self.features = []
        self.reduced_features = []
        self.imagesize = 256
        self.patchcore = patchcore
        # self.backbone = YuShuanPatch(
        #     backbone_name="wide_resnet50_2", 
        #     imagesize=self.imagesize, 
        #     device="cuda", 
        #     normalize=False,
        #     mean=False,
        # )
        self.backbone = load_backbone("resnet18", ["layer2","layer3"], imagesize=256, device="cuda")
    
    def mean_shift(self, feats, b=None):
        meanshift = MeanShift(n_jobs=-1) if b is None else MeanShift(bandwidth=b,n_jobs=-1)
        cluster_labels = meanshift.fit_predict(feats)
        cluster_centers = meanshift.cluster_centers_
        return cluster_labels, cluster_centers
    
    def dbscan(self, eps=0.5, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = db.fit_predict(self.features)
        return cluster_labels

    def get_imgid_instdirs(self, all_instances):
        img_ids = set()
        instance_dirs = set()
        for instance in all_instances:
            instance_dir, img_id = os.path.dirname(instance), os.path.splitext(os.path.basename(instance))[0].split("_")[0]
            img_ids.add(img_id)
            instance_dirs.add(instance_dir)
        return img_ids, instance_dirs

    def get_normal_features(self, all_instances):
        instance_dataset = ImageDataset(self.imagesize, all_instances, split="train")
        instance_dataloader = torch.utils.data.DataLoader(
            instance_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        ### original patchcore
        feats = self.patchcore.embed(instance_dataloader)
        self.features = np.concatenate(feats, axis=0)
        
        return self.features

if __name__=="__main__":
	pass