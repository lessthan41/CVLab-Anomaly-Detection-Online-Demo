"""Sample evaluation script for track 1."""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn

# from anomalib.data import MVTec
# from anomalib.metrics import F1Max

import timm
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
from csad.vis_histogram import histogram,patch_histogram,vis_histogram
from .models.segment2.model import Segmentor


class Padding2Resize():
    def __init__(self, pad_l, pad_t, pad_r, pad_b):
        self.pad_l = pad_l
        self.pad_t = pad_t
        self.pad_r = pad_r
        self.pad_b = pad_b

    def __call__(self,image,target_size,mode='bilinear'):
        if len(image.shape) == 3:
            shape=3
            image = image[None,:,:,:]
        # B,C,H,W
        if self.pad_b == 0:
            image = image[:,:,self.pad_t:]
        else:
            image = image[:,:,self.pad_t:-self.pad_b]
        if self.pad_r == 0:
            image = image[:,:,:,self.pad_l:]
        else:
            image = image[:,:,:,self.pad_l:-self.pad_r]
        image = torch.nn.functional.interpolate(image, size=(target_size,target_size), mode=mode)
        if shape == 3:
            return image[0]
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



class CSAD(nn.Module):
    def __init__(self,category):
        super().__init__()
        self.category = category
        # models
        self.encoder = timm.create_model('wide_resnet50_2',
                                            pretrained=True,
                                            features_only=True,
                                            out_indices=[1,2,3]).eval().cuda()

        self.segmentor = torch.load(f'/home/tokichan/JL/csad/ckpt/segmentor_{category}_256.pth').eval().cuda()
        self.num_classes = self.segmentor.fc2.conv3.out_channels-1
        LGST_ckpt = torch.load(f'/home/tokichan/JL/csad/ckpt/best_{category}.pth')
        self.teacher = LGST_ckpt['teacher'].eval().cuda()
        self.teacher.encoder = self.encoder
        self.student = LGST_ckpt['student'].eval().cuda()
        self.autoencoder = LGST_ckpt['autoencoder'].eval().cuda()
        
        # load params
        self.teacher_mean = LGST_ckpt['teacher_mean'].cuda()
        self.teacher_std = LGST_ckpt['teacher_std'].cuda()
        self.q_ae_end = LGST_ckpt['q_ae_end'].cuda()
        self.q_ae_start = LGST_ckpt['q_ae_start'].cuda()
        self.q_st_end = LGST_ckpt['q_st_end'].cuda()
        self.q_st_start = LGST_ckpt['q_st_start'].cuda()

        # transforms
        self.LGST_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

        self.padding, self.pad2resize = get_padding_functions([256,256],256)
        
        # load normal images
        # self.images = images
        
        # load normal hists
        self.normal_hists = np.load(f'/home/tokichan/JL/csad/ckpt/normal_hists_{category}.npy')
        # load normal patchhist
        self.normal_patchhists = np.load(f'/home/tokichan/JL/csad/ckpt/normal_patchhists_{category}.npy')
        # load normal segmaps
        self.normal_segmaps = np.load(f'/home/tokichan/JL/csad/ckpt/normal_segmaps_{category}.npy')
        
        
    def LGST_forward(self,image):
        image = self.LGST_transform(image).cuda().unsqueeze(0)
        teacher_output = self.teacher(image)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        st_student_output,ae_student_output = self.student(image)

        autoencoder_output = self.autoencoder(image)
        
        map_st = torch.mean((teacher_output - st_student_output)**2,
                            dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - ae_student_output)**2,
                            dim=1, keepdim=True)
        
        if self.q_st_start is not None:
            if (self.q_st_end - self.q_st_start)>1e-6:
                map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
            else:
                print(f'warn: self.q_st_end:{self.q_st_end} - self.q_st_start:{self.q_st_start} < 1e-6')
        if self.q_ae_start is not None:
            if (self.q_ae_end - self.q_ae_start)>1e-6:
                map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
            else:
                print(f'warn: self.q_ae_end:{self.q_ae_end} - self.q_ae_start:{self.q_ae_start} < 1e-6')

        map_combined = map_st + map_ae
        map_combined = torch.nn.functional.interpolate(map_combined, size=(256,256), mode='bilinear', align_corners=False)
        return map_combined.cpu().detach().numpy()[0,0]

    def patcchhist_forward(self,image):
        pad_image = self.seg_transform(self.padding(image)).cuda().unsqueeze(0)
        feat = self.encoder(pad_image)
        segmap = self.segmentor(feat)
        segmap = segmap.argmax(dim=1)
        segmap = self.pad2resize(segmap,256)[0]
        segmap = segmap.cpu().detach().numpy()

        anomaly_map = vis_histogram(segmap,self.normal_hists,self.normal_patchhists,self.normal_segmaps,self.num_classes,category=self.category)
        return anomaly_map
        
        
        
        
    def forward(self, img):
        LGST_map = self.LGST_forward(img)
        patchhist_map = self.patcchhist_forward(img)
        
        anomaly_map = LGST_map + patchhist_map*0.0
        anomaly_score = np.max(LGST_map) + np.max(patchhist_map)
        
        return anomaly_map, anomaly_score


if __name__ == "__main__":
    images = []
    csad = CSAD(images,category='breakfast_box')
    anomaly_map, anomaly_score = csad(Image.fromarray(np.random.randint(0,255,(500,256,3)).astype(np.uint8)))