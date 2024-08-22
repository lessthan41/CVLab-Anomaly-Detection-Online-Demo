import os
import os
import PIL
import glob
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def image_transform(image, image_size):
	transform = transforms.Compose([
		transforms.Resize([image_size,image_size]),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])
	image = transform(image)
	return image

def update_instances(img_id, instance_dirs):
    """get instances in img_id from instance_dir and sort them"""
    instances = []
    for instance_dir in instance_dirs:
        instances += glob.glob(os.path.join(instance_dir, "*.*"))
    _instances = [i for i in instances if any(os.path.basename(i).split("_")[0] == j for j in img_id)]
    _instances.sort()
    return _instances.copy()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagesize=256,
        img_id=None,
        instance_dirs=None,
        mask_path=None,
        interval=30,
        **kwargs,
    ):
        super().__init__()
        self.data_to_iterate = update_instances([img_id], instance_dirs)
        self.transform_img = [
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

        self.imagesize = (3, imagesize, imagesize)
        self.rotations = int(360 / interval)
        self.degree = interval
        self.mask = mask_path
        if self.mask is not None:
            self.mask = self.transform_mask(PIL.Image.open(self.mask).convert("L"))
            self.mask = self.mask > 0.5


    def __getitem__(self, idx):
        img_id = idx // self.rotations
        degree = self.degree * (idx % self.rotations)
        
        image_path = self.data_to_iterate[img_id]
        image = PIL.Image.open(image_path).convert("RGB")
        image = transforms.functional.rotate(image, degree, interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.transform_img(image)

        if self.mask is not None:
            mask = transforms.functional.rotate(self.mask, degree, interpolation=transforms.InterpolationMode.NEAREST)
            image = image * mask

        return {
            "image": image,
            "degree": degree,
        }

    def __len__(self):
        return len(self.data_to_iterate) * self.rotations

class InstanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagesize=256,
        i_path=None,
        interval=30,
        **kwargs,
    ):
        super().__init__()
        self.i_path = i_path
        self.transform_img = [
            transforms.Resize([imagesize,imagesize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
        self.imagesize = (3, imagesize, imagesize)
        self.rotations = int(360 / interval)
        self.degree = interval

    def __getitem__(self, idx):
        degree = self.degree * (idx % self.rotations)
        image = PIL.Image.open(self.i_path).convert("RGB")
        image = transforms.functional.rotate(image, degree, interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.transform_img(image)

        return {
            "image": image,
            "degree": degree,
        }

    def __len__(self):
        return self.rotations

class InstancesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagesize=256,
        path=None,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.transform_img = [
            transforms.Resize([imagesize,imagesize]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.path[idx]).convert("RGB")
        image = self.transform_img(image)

        return {
            "image": image,
            "image_path": self.path[idx],
        }

    def __len__(self):
        return len(self.path)

if __name__=="__main__":
    inst = InstanceDataset(
        i_path="/home/anomaly/data/segment/output/visa/instance/capsules/train/good/023_16.JPG",
        imagesize=256,
        interval=20,
    )
    dataloader = torch.utils.data.DataLoader(
        inst,
        batch_size=8,
        shuffle=False,
    )
    for data in dataloader:
        continue