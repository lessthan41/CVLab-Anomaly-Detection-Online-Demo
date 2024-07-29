import torch
import torchvision.transforms as transforms
import numpy as np
import timm
import cv2
from sklearn.preprocessing import minmax_scale
# from data_loader import MVTecLOCODataset
import yaml
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
# import tifffile
import os
import scipy

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def cal_component_score(hist,normal_hist,patch_num):
    """
    Args:
        hist: 1xN array
        normal_hist: 1xN array
        side_patch_num: int
    """
    small_thresh = 50
    hist = np.reshape(hist,(patch_num,-1))+1
    normal_hist = np.reshape(normal_hist,(patch_num,-1))+1
    diff = np.abs(hist - normal_hist)
    normal_hist[normal_hist<small_thresh] = 256**2
    diff = diff / normal_hist
    if patch_num == 1:
        score = diff
        more_or_less = np.where(hist>normal_hist,1,-1)
    else:
        score = np.max(diff,axis=0,keepdims=True)
        more_or_less = np.ones((1,diff.shape[1])).astype(np.int32)*-1
    
    return score[0],more_or_less[0]



def draw_anomaly_map_patchhist(com_score,pred_segmap,more_or_less,normal_hist,normal_segmap,num_classes,threshold=0.7):
    anomaly_map = np.zeros((256,256)).astype(np.float32)
    # com_score = minmax_scale(com_score)

    # add missing component
    for i in range(com_score.shape[0]):
        if com_score[i]>threshold:
            # missing component
            anomaly_map[normal_segmap==i+1] = com_score[i]
            pred_segmap[normal_segmap==i+1] = normal_segmap[normal_segmap==i+1]
    # new_hist = histogram(pred_segmap,num_classes)
    new_patch_hist = patch_histogram(pred_segmap,num_classes)
    new_com_score,new_more_or_less = cal_component_score(new_patch_hist,normal_hist,4)

    # # mark additional component
    for i in range(com_score.shape[0]):
        if new_more_or_less[i]==1 and new_com_score[i]>threshold:
            # more component
            # anomaly_map[pred_segmap==i+1] += com_score[i]
            anomaly_map[pred_segmap==i+1] = np.maximum(anomaly_map[pred_segmap==i+1],new_com_score[i])

    return anomaly_map

def draw_anomaly_map(com_score,pred_segmap,more_or_less,normal_hist,normal_segmap,threshold=0.4):
    anomaly_map = np.zeros((256,256)).astype(np.float32)
    # com_score = minmax_scale(com_score)

    # add missing component
    for i in range(com_score.shape[0]):
        if more_or_less[i]==-1 and com_score[i]>threshold:
            # missing component
            anomaly_map[normal_segmap==i+1] = com_score[i]
            pred_segmap[normal_segmap==i+1] = normal_segmap[normal_segmap==i+1]
        elif more_or_less[i]==1 and com_score[i]>threshold:
            anomaly_map[pred_segmap==i+1] = np.maximum(anomaly_map[pred_segmap==i+1],com_score[i])

    # new_hist = histogram(pred_segmap,num_classes)
    # # new_patch_hist = patch_histogram(pred_segmap,num_classes)
    # new_com_score,new_more_or_less = cal_component_score(new_hist,normal_hist,1)

    # # mark additional component
    # for i in range(com_score.shape[0]):
    #     if new_more_or_less[i]==1 and new_com_score[i]>threshold:
    #         # more component
    #         # anomaly_map[pred_segmap==i+1] += com_score[i]
            
    return anomaly_map

def patch_histogram(label_map, num_classes):
    h,w = label_map.shape
    p1 = label_map[0:h//2,0:w//2]
    p2 = label_map[0:h//2,w//2:w]
    p3 = label_map[h//2:h,0:w//2]
    p4 = label_map[h//2:h,w//2:w]
    # p1 = label_map[0:h//2,:]
    # p2 = label_map[h//2:h,:]
    # p3 = label_map[:,0:w//2]
    # p4 = label_map[:,w//2:w]
    hists = [histogram(p,num_classes) for p in [p1,p2,p3,p4]]
    hists = np.hstack(hists)
    return hists


def histogram(label_map,num_classes):
    hist = np.zeros(num_classes)
    for i in range(1,num_classes+1): # not include background
        hist[i-1] = (label_map == i).sum()
    # hist = hist / label_map.size
    return hist 

def de_normalize(tensor):
    # tensor: (B,C,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return tensor
color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]

def vis_histogram(segmap,normal_hists,normal_patchhists,normal_segmaps,num_classes,category):
    hist = histogram(segmap,num_classes)
    # global_feat = torch.mean(feat[2].squeeze(),dim=(1,2))

    # find closest normal image and its segmap
    dist = -torch.sum((torch.from_numpy(hist) - torch.from_numpy(np.array(normal_hists)))**2,dim=1)
    # smallest k distance index
    indices = torch.topk(dist,5).indices

    k_normal_hists = [normal_hists[idx] for idx in indices]
    k_normal_patchhists = [normal_patchhists[idx] for idx in indices]
    k_normal_segmaps = [normal_segmaps[idx] for idx in indices]
    normal_hist = np.mean(k_normal_hists,axis=0)
    normal_patchhist = np.mean(k_normal_patchhists,axis=0)
    normal_segmap = k_normal_segmaps[0]

    # normal_segmap = normal_segmaps[min_idx]
    # normal_hist = normal_hists[min_idx]
    # normal_patchhist = normal_patchhists[min_idx]


    
    
    # visualize segmap
    # color_segmap = np.zeros((256,256,3))
    # for i in range(1,num_classes+1):
    #     color_segmap[segmap==i] = color_list[i-1][::-1]
    # color_segmaps.append(color_segmap)
    # cv2.imshow('segmap',color_segmap)
    # cv2.waitKey(0)

    hist = histogram(segmap,num_classes)
    
    patch_hist = patch_histogram(segmap,num_classes)

    com_score,more_or_less = cal_component_score(hist,normal_hist,1)
    anomaly_map = draw_anomaly_map(
        com_score= com_score,
        pred_segmap=segmap,
        normal_hist=normal_hist,
        normal_segmap=normal_segmap,
        more_or_less=more_or_less,)
    com_score_patch,more_or_less_patch = cal_component_score(patch_hist,normal_patchhist,4)
    anomaly_map_patch = draw_anomaly_map_patchhist(
        com_score= com_score_patch,
        pred_segmap=segmap,
        normal_hist=normal_patchhist,
        normal_segmap=normal_segmap,
        more_or_less=more_or_less_patch,
        num_classes=num_classes)
    
    patchhist_ratio = 0 if category == "screw_bag" or category == "breakfast_box" else 0.5
    return anomaly_map + anomaly_map_patch*patchhist_ratio

    



if __name__ == "__main__":
    categories = ["screw_bag",'breakfast_box',"splicing_connectors","juice_bottle","pushpins",]
    for category in categories:
        image_size = 256
        encoder = timm.create_model('wide_resnet50_2'
                                                    ,pretrained=True,
                                                    features_only=True,
                                                    out_indices=[1,2,3])

        for name,param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.cuda()

        segmentor = torch.load(f"./ckpt/segmentor_{category}_{image_size}.pth")
        segmentor.cuda()
        segmentor.eval()
        

        dataset = MVTecLOCODataset(
            root="C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/",
            category=category,
            image_size=image_size,
            phase='train'
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        val_dataset = MVTecLOCODataset(
            root="C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/",
            category=category,
            image_size=image_size,
            phase='eval'
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        test_dataset = MVTecLOCODataset(
            root="C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/",
            category=category,
            image_size=image_size,
            phase='test'
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        num_classes = segmentor.fc2.conv3.out_channels-1

        pad2resize = dataset.pad2resize

        # get normal hist
        normal_hists = []
        normal_images = []
        normal_segmaps = []
        normal_patchhists = []
        normal_feats = []
        for i, sample in tqdm.tqdm(enumerate(dataloader),desc='build patch histogram'):
            
            with torch.no_grad():
                image = sample['pad_image']
                normal_images.append(image)
                image.cuda()
                feat = encoder(image)

                global_feat = torch.mean(feat[2].squeeze(),dim=(1,2))
                normal_feats.append(global_feat)
                segmap = segmentor(feat)
                segmap = pad2resize(segmap,target_size=256)
                segmap = torch.argmax(segmap, dim=1).cpu().numpy()[0]

                # visualize segmap
                # color_segmap = np.zeros((256,256,3))
                # for i in range(num_classes):
                #     color_segmap[segmap==i] = np.random.rand(3)
                # cv2.imshow('segmap',color_segmap)
                # cv2.waitKey(0)


                normal_segmaps.append(segmap)

                hist = histogram(segmap,num_classes)
                normal_hists.append(hist)
                patch_hist = patch_histogram(segmap,num_classes)
                normal_patchhists.append(patch_hist)

        global_feats = torch.stack(normal_feats)

        print()
        # mean_normal_hist = np.mean(normal_hists,axis=0)
        # mean_normal_hist = scipy.stats.trim_mean(normal_hists,0.2,axis=0)
        # mean_normal_patchhist = np.mean(normal_patchhists,axis=0)
        # mean_normal_patchhist = scipy.stats.trim_mean(normal_patchhists,0.2,axis=0)
        
        anomaly_maps = []
        color_segmaps = []
        for i, sample in tqdm.tqdm(enumerate(test_dataloader),desc='build patch histogram'):
            # if i<122+74:
            #     continue
            with torch.no_grad():
                image = sample['pad_image']
                image.cuda()
                feat = encoder(image)
                segmap = segmentor(feat)
                segmap = pad2resize(segmap,target_size=256)
                segmap = torch.argmax(segmap, dim=1).cpu().numpy()[0]
                hist = histogram(segmap,num_classes)
                # global_feat = torch.mean(feat[2].squeeze(),dim=(1,2))

                # find closest normal image and its segmap
                dist = -torch.sum((torch.from_numpy(hist) - torch.from_numpy(np.array(normal_hists)))**2,dim=1)
                # smallest k distance index
                indices = torch.topk(dist,5).indices

                k_normal_hists = [normal_hists[idx] for idx in indices]
                k_normal_patchhists = [normal_patchhists[idx] for idx in indices]
                k_normal_segmaps = [normal_segmaps[idx] for idx in indices]
                normal_hist = np.mean(k_normal_hists,axis=0)
                normal_patchhist = np.mean(k_normal_patchhists,axis=0)
                normal_segmap = k_normal_segmaps[0]

                # normal_segmap = normal_segmaps[min_idx]
                # normal_hist = normal_hists[min_idx]
                # normal_patchhist = normal_patchhists[min_idx]


                
                
                # visualize segmap
                color_segmap = np.zeros((256,256,3))
                for i in range(1,num_classes+1):
                    color_segmap[segmap==i] = color_list[i-1][::-1]
                color_segmaps.append(color_segmap)
                # cv2.imshow('segmap',color_segmap)
                # cv2.waitKey(0)

                hist = histogram(segmap,num_classes)
                
                patch_hist = patch_histogram(segmap,num_classes)

                com_score,more_or_less = cal_component_score(hist,normal_hist,1)
                anomaly_map = draw_anomaly_map(
                    com_score= com_score,
                    pred_segmap=segmap,
                    normal_hist=normal_hist,
                    normal_segmap=normal_segmap,
                    more_or_less=more_or_less,)
                com_score_patch,more_or_less_patch = cal_component_score(patch_hist,normal_patchhist,4)
                anomaly_map_patch = draw_anomaly_map_patchhist(
                    com_score= com_score_patch,
                    pred_segmap=segmap,
                    normal_hist=normal_patchhist,
                    normal_segmap=normal_segmap,
                    more_or_less=more_or_less_patch,)
                


                # save anomaly map
                name = sample['name'][0]
                img_type = sample['type'][0]
                save_path = f"./anomaly_map/patchhist/{category}/test/{img_type}/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                patchhist_ratio = 0 if category == "screw_bag" or category == "breakfast_box" else 0.5
                tifffile.imwrite(save_path+name+'.tiff',(anomaly_map + anomaly_map_patch*patchhist_ratio))

                
                # anomaly_map = anomaly_map / (np.max(anomaly_map)+1)
                # anomaly_map = np.reshape(anomaly_map*255,(256,256)).astype(np.uint8)
                # anomaly_map = cv2.applyColorMap(anomaly_map,cv2.COLORMAP_JET)
                # cv2.cvtColor(anomaly_map,cv2.COLOR_BGR2RGB,anomaly_map)
                # anomaly_map_patch_patch = anomaly_map_patch / np.max(anomaly_map_patch)
                # anomaly_map_patch = np.reshape(anomaly_map_patch*255,(256,256)).astype(np.uint8)
                # anomaly_map_patch = cv2.applyColorMap(anomaly_map_patch,cv2.COLORMAP_JET)
                # cv2.cvtColor(anomaly_map_patch,cv2.COLOR_BGR2RGB,anomaly_map_patch)
                # # anomaly_map = cv2.merge((anomaly_map,anomaly_map,anomaly_map))
                # image = de_normalize(image)
                # image = pad2resize(image,target_size=256)
                # image = image.squeeze().cpu().numpy().transpose(1,2,0)
                # image = (image*255).astype(np.uint8)
                # vis_image = np.hstack((image,color_segmap,anomaly_map,anomaly_map_patch)).astype(np.uint8)
                # plt.imshow(vis_image)
                # plt.show()
                # print()

                # anomaly_maps.append(anomaly_map)


    
            
    