import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from einops import rearrange
from scipy.ndimage import binary_fill_holes

class ComponentFeatureExtractor():
    """
    "com_config":{
            'reduce_feature': 'mean',
            'in_dim': 1024,
            'proj_dim': 8,
            'bins': 20,
            'dim_reduction': False,
            'normalize': True,
            'image_size': 256,
            'transform': transform,
        }
    """
    def __init__(self,config,model=None):
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = config["transform"]
        if self.model is not None:
            self.model.to(self.device)

    def crop_by_mask(self,image,mask):
        if image.shape[-1] == 3:
            mask = cv2.merge([mask,mask,mask])
        return np.where(mask!=0,image,0)
    
    def align_with_relative_scale(self,masks,target_size,image):
        """
            resize object and record scale position
            target_size: Int
            output: List[cropped_masks,cropped_images,center_positions,scales]
        """
        cropped_masks = list()
        cropped_images = list()
        center_positions = list()
        scales = list()

        i=0
        max_diagonal = 0
        for mask in masks:
            cnt,_ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key = cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            p1,p2 = box[0], box[2]

            diagonal = np.linalg.norm((p1-p2),ord=2)
            max_diagonal = max(max_diagonal,diagonal)
        
        for mask in masks:
            # crop image by cropped_mask
            croped_image = self.crop_by_mask(image, mask)

            cnt,_ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key = cv2.contourArea)

            # fill holes in the mask
            mask = cv2.drawContours(np.zeros_like(mask), [c], -1, (255,255,255), -1)

            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            p1,p2 = box[0], box[2]

            diagonal = np.linalg.norm((p1-p2),ord=2)


            # resize image and mask to target_size if diagonal is max_diagonal
            scale = (target_size/(diagonal+1)) * ((diagonal+1)/max_diagonal)


            cropped_mask = mask[y:y+h, x:x+w]
            croped_image = croped_image[y:y+h, x:x+w]
            temp_size = (np.array(cropped_mask.shape) * scale).astype(np.int32)
            temp_size = np.where(temp_size>target_size,target_size,temp_size)

            cropped_mask = cv2.resize(cropped_mask,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            croped_image = cv2.resize(croped_image,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            cropped_mask[cropped_mask>128] = 255
            cropped_mask[cropped_mask<=128] = 0

            # padding
            padw = int((target_size - cropped_mask.shape[1])//2)
            padh = int((target_size - cropped_mask.shape[0])//2)
            cropped_mask = cv2.copyMakeBorder(cropped_mask,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))
            croped_image = cv2.copyMakeBorder(croped_image,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))

            if cropped_mask.shape[0] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
            if cropped_mask.shape[1] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))

            #print(cropped_mask.shape)
            # normalized center position of the mask object
            # range:[0,1]
            center_position = [(x+w/2)/mask.shape[1], (y+h/2)/mask.shape[0]]
            
            cropped_masks.append(cropped_mask)
            cropped_images.append(croped_image)
            center_positions.append(center_position)
            scales.append(scale)

            i+=1

        # # visualize cropped images
        # import matplotlib.pyplot as plt
        # vis1 = np.vstack(cropped_images[0:3])
        # vis2 = np.vstack(cropped_images[3:6])
        # vis3 = np.vstack(cropped_images[6:9])
        # vis = np.hstack([vis1,vis2,vis3])
        # plt.imshow(vis)
        # plt.show()


        return [cropped_masks,cropped_images,center_positions,scales]

    def align(self,masks,target_size,image):
        """
            resize object and record scale position
            target_size: Int
            output: List[cropped_masks,cropped_images,center_positions,scales]
        """
        cropped_masks = list()
        cropped_images = list()
        center_positions = list()
        scales = list()

        i=0
        for mask in masks:
            
            # crop image by cropped_mask
            croped_image = self.crop_by_mask(image, mask)

            cnt,_ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key = cv2.contourArea)

            # fill holes in the mask
            mask = cv2.drawContours(np.zeros_like(mask), [c], -1, (255,255,255), -1)

            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            p1,p2 = box[0], box[2]

            diagonal = np.linalg.norm((p1-p2),ord=2)
            scale = target_size/(diagonal+1)
            cropped_mask = mask[y:y+h, x:x+w]
            croped_image = croped_image[y:y+h, x:x+w]
            temp_size = (np.array(cropped_mask.shape) * scale).astype(np.int32)
            temp_size = np.where(temp_size>target_size,target_size,temp_size)
            temp_size = np.where(temp_size<=0,1,temp_size)

            cropped_mask = cv2.resize(cropped_mask,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            croped_image = cv2.resize(croped_image,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            cropped_mask[cropped_mask>128] = 255
            cropped_mask[cropped_mask<=128] = 0

            # padding
            padw = int((target_size - cropped_mask.shape[1])//2)
            padh = int((target_size - cropped_mask.shape[0])//2)
            cropped_mask = cv2.copyMakeBorder(cropped_mask,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))
            croped_image = cv2.copyMakeBorder(croped_image,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))

            if cropped_mask.shape[0] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
            if cropped_mask.shape[1] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))

            #print(cropped_mask.shape)
            # normalized center position of the mask object
            # range:[0,1]
            center_position = [(x+w/2)/mask.shape[1], (y+h/2)/mask.shape[0]]
            
            cropped_masks.append(cropped_mask)
            cropped_images.append(croped_image)
            center_positions.append(center_position)
            scales.append(scale)

            # cv2.imwrite(f"/home/anomaly/DRAEM/pack/capsule/mask2/{i}.jpg",cropped_mask)
            i+=1
        return [cropped_masks,cropped_images,center_positions,scales]
    
    
    def rot_images(self,images,num_angles=20):
        images = [Image.fromarray(image) for image in images]
        result = list()
        for image in images:
            rotated_image = torch.stack([self.transform(image.rotate(angle,Image.Resampling.BILINEAR)) for angle in np.linspace(0,360,num_angles)])
            result.append(rotated_image)
        result = torch.cat(result,dim=0)
        return result
    
    def fill_holes(self,masks):
        result = list()
        for mask in masks:
            mask = binary_fill_holes(np.where(mask!=0,1,0))
            mask = np.where(mask!=0,255,0)
            mask = mask.astype(np.uint8)
            result.append(mask)
        return result


        
    def compute_component_feature(self,image,masks):
        """
        image: [256,256,3] contains a component, value range [0,255]
        mask: [N,256,256] contains a component mask, value range [0,255]
        output: List[feature_dict]
        """
        # area feature (from ComAD)
        component_features = list()
        aligned_info = self.align(masks=masks,target_size=65,image=image)
        # scaled_aligned_info = self.align_with_relative_scale(masks=masks,target_size=256,image=image)
        # scaled_aligned_images = scaled_aligned_info[1]
        # aligned_masks = self.fill_holes(aligned_info[0])
        aligned_masks = aligned_info[0]
        aligned_images = aligned_info[1]


        for i,mask in enumerate(masks):
            feature = dict()
            # cv2.imshow('mask',mask)
            # cv2.waitKey(0)

            area = np.sum(mask!=0)
            feature['area'] = np.array([area])/mask.size

            # color feature (from ComAD)
            image_lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB) # [256,256,3]
            color_sum_a = image_lab[:,:,1].astype(np.float32)
            color_sum_b = image_lab[:,:,2].astype(np.float32)
            color_div = (color_sum_b/(color_sum_a+0.0000001))*(color_sum_b/(color_sum_a+0.0000001))
            color_div = color_div * np.where(mask!=0,1,0)
            color_value = np.sum(color_div)/(area+0.0000001)
            feature['color'] = np.array([color_value])

            # position feature
            # center of bounding box (normalized)
            x,y,w,h = cv2.boundingRect(mask)
            x = x+w/2
            y = y+h/2
            center_x = x/mask.shape[1]
            center_y = y/mask.shape[0]
            position = np.array([center_x,center_y])
            feature['position'] = position

            # shape feature
            # hu moments
            moments = cv2.moments(aligned_masks[i])
            hu_moments = cv2.HuMoments(moments)
            # Log scale hu moments 
            for j in range(0,7):
                if hu_moments[j] != 0:
                    hu_moments[j] = -1 * np.copysign(1.0, hu_moments[j]) * np.log10(abs(hu_moments[j]))
                else:
                    hu_moments[j] = 0
            # hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
            # hu_moments_log = np.where(np.isnan(hu_moments_log), 0, hu_moments_log)
            hu_moments = np.squeeze(hu_moments)
            feature['shape'] = hu_moments

            component_features.append(feature)
            assert np.isnan(hu_moments).sum() == 0 , "component feature contains nan"

            feature['scale'] = aligned_info[3][i]

        # CNN feature
        # crop image and resize to 64x64
        num_rotation = 60
        
        if self.model is not None:
            # aligned_shapes = [np.stack([mask,mask,mask],axis=2) for mask in aligned_masks]
            # rotated_shapes = self.rot_images(aligned_shapes,num_angles=num_rotation)
            rotated_images = self.rot_images(aligned_images,num_angles=num_rotation)
            #scaled_rotated_images = self.rot_images(scaled_aligned_images,num_angles=num_rotation)
            #################################
            # rotated_shapes1 = self.rot_images(aligned_shapes,num_angles=1)
            # rotated_images1 = self.rot_images(aligned_images,num_angles=1)
            # rotated_shapes15 = self.rot_images(aligned_shapes,num_angles=15)
            # rotated_images15 = self.rot_images(aligned_images,num_angles=15)
            # rotated_shapes30 = self.rot_images(aligned_shapes,num_angles=30)
            # rotated_images30 = self.rot_images(aligned_images,num_angles=30)

            with torch.no_grad():
                #shape_outputs = self.model(rotated_shapes.to(self.device))[0]
                image_outputs = self.model(rotated_images.to(self.device))[0]
                # scaled_image_outputs = self.model(scaled_rotated_images.to(self.device))[0]
                #################################
                # shape_outputs1 = self.model(rotated_shapes1.to(self.device))[0]
                # image_outputs1 = self.model(rotated_images1.to(self.device))[0]
                # shape_outputs15 = self.model(rotated_shapes15.to(self.device))[0]
                # image_outputs15 = self.model(rotated_images15.to(self.device))[0]
                # shape_outputs30 = self.model(rotated_shapes30.to(self.device))[0]
                # image_outputs30 = self.model(rotated_images30.to(self.device))[0]

            #del rotated_shapes
            del rotated_images
            ########################
            # del rotated_shapes1
            # del rotated_images1
            # del rotated_shapes15
            # del rotated_images15
            # del rotated_shapes30
            # del rotated_images30
            # torch.cuda.empty_cache()
            
            # shape_outputs = rearrange(shape_outputs,'(N R) C H W -> N R C H W',R=num_rotation)
            # shape_outputs = torch.mean(shape_outputs,dim=[3,4],keepdim=False) # average over spatial dimension
            # shape_outputs = torch.mean(shape_outputs,dim=1,keepdim=False) # average over rotation

            image_outputs = rearrange(image_outputs,'(N R) C H W -> N R C H W',R=num_rotation)
            image_outputs = torch.mean(image_outputs,dim=[3,4],keepdim=False) # average over spatial dimension
            image_outputs = torch.mean(image_outputs,dim=1,keepdim=False) # average over rotation\
            
            # scaled_image_outputs = rearrange(scaled_image_outputs,'(N R) C H W -> N R C H W',R=num_rotation)
            # scaled_image_outputs = torch.mean(scaled_image_outputs,dim=[3,4],keepdim=False) # average over spatial dimension
            # scaled_image_outputs = torch.mean(scaled_image_outputs,dim=1,keepdim=False) # average over rotation\

            
            ##########################################
            # shape_outputs1 = rearrange(shape_outputs1,'(N R) C H W -> N R C H W',R=1)
            # shape_outputs1 = torch.mean(shape_outputs1,dim=[3,4],keepdim=False) # average over spatial dimension
            # shape_outputs1 = torch.mean(shape_outputs1,dim=1,keepdim=False) # average over rotation

            # image_outputs1 = rearrange(image_outputs1,'(N R) C H W -> N R C H W',R=1)
            # image_outputs1 = torch.mean(image_outputs1,dim=[3,4],keepdim=False) # average over spatial dimension
            # image_outputs1 = torch.mean(image_outputs1,dim=1,keepdim=False) # average over rotation

            # shape_outputs15 = rearrange(shape_outputs15,'(N R) C H W -> N R C H W',R=15)
            # shape_outputs15 = torch.mean(shape_outputs15,dim=[3,4],keepdim=False) # average over spatial dimension
            # shape_outputs15 = torch.mean(shape_outputs15,dim=1,keepdim=False) # average over rotation

            # image_outputs15 = rearrange(image_outputs15,'(N R) C H W -> N R C H W',R=15)
            # image_outputs15 = torch.mean(image_outputs15,dim=[3,4],keepdim=False) # average over spatial dimension
            # image_outputs15 = torch.mean(image_outputs15,dim=1,keepdim=False) # average over rotation

            # shape_outputs30 = rearrange(shape_outputs30,'(N R) C H W -> N R C H W',R=30)
            # shape_outputs30 = torch.mean(shape_outputs30,dim=[3,4],keepdim=False) # average over spatial dimension
            # shape_outputs30 = torch.mean(shape_outputs30,dim=1,keepdim=False) # average over rotation

            # image_outputs30 = rearrange(image_outputs30,'(N R) C H W -> N R C H W',R=30)
            # image_outputs30 = torch.mean(image_outputs30,dim=[3,4],keepdim=False) # average over spatial dimension
            # image_outputs30 = torch.mean(image_outputs30,dim=1,keepdim=False) # average over rotation





        
            for i in range(len(component_features)):
                #component_features[i]['cnn_shape'] = shape_outputs[i].cpu().numpy()
                component_features[i]['cnn_image'] = image_outputs[i].cpu().numpy()
                # component_features[i]['cnn_image_scaled'] = scaled_image_outputs[i].cpu().numpy()
                # component_features[i]['cnn_shape1'] = shape_outputs1[i].cpu().numpy()
                # component_features[i]['cnn_image1'] = image_outputs1[i].cpu().numpy()
                # component_features[i]['cnn_shape15'] = shape_outputs15[i].cpu().numpy()
                # component_features[i]['cnn_image15'] = image_outputs15[i].cpu().numpy()
                # component_features[i]['cnn_shape30'] = shape_outputs30[i].cpu().numpy()
                # component_features[i]['cnn_image30'] = image_outputs30[i].cpu().numpy()

            #del shape_outputs
            del image_outputs
            ################################
            # del shape_outputs1
            # del image_outputs1
            # del shape_outputs15
            # del image_outputs15
            # del shape_outputs30
            # del image_outputs30
            # torch.cuda.empty_cache()
        # #####################################
        # visualize component feature
        # import matplotlib.pyplot as plt
        # import sklearn.metrics.pairwise as pw
        # shape60 = np.array([feature['cnn_shape'] for feature in component_features])
        # shape1 = np.array([feature['cnn_shape1'] for feature in component_features])
        # shape15 = np.array([feature['cnn_shape15'] for feature in component_features])
        # shape30 = np.array([feature['cnn_shape30'] for feature in component_features])

        # image1 = np.array([feature['cnn_image1'] for feature in component_features])
        # image15 = np.array([feature['cnn_image15'] for feature in component_features])
        # image30 = np.array([feature['cnn_image30'] for feature in component_features])
        # image60 = np.array([feature['cnn_image'] for feature in component_features])
        
        # fig = plt.figure(dpi=150)

        # ax = fig.add_subplot(151)
        # pair_dist1 = pw.pairwise_distances(shape1,metric='euclidean')
        # ax.set_title('shape1')
        # ax.imshow(pair_dist1.max()-pair_dist1)

        # ax = fig.add_subplot(152)
        # pair_dist15 = pw.pairwise_distances(shape15,metric='euclidean')
        # ax.set_title('shape15')
        # ax.imshow(pair_dist15.max()-pair_dist15)

        # ax = fig.add_subplot(153)
        # pair_dist30 = pw.pairwise_distances(shape30,metric='euclidean')
        # ax.set_title('shape30')
        # ax.imshow(pair_dist30.max()-pair_dist30)

        # ax = fig.add_subplot(154)
        # pair_dist60 = pw.pairwise_distances(shape60,metric='euclidean')
        # ax.set_title('shape60')
        # ax.imshow(pair_dist60.max()-pair_dist60)

        # ax = fig.add_subplot(155)
        # images = np.vstack(aligned_shapes)
        # ax.imshow(images)
        # plt.savefig('shape.png')
        # plt.close()

        # #####################################
        # fig = plt.figure(dpi=150)

        # ax = fig.add_subplot(151)
        # pair_dist1 = pw.pairwise_distances(image1,metric='euclidean')
        # ax.set_title('image1')
        # ax.imshow(pair_dist1.max()-pair_dist1)

        # ax = fig.add_subplot(152)
        # pair_dist15 = pw.pairwise_distances(image15,metric='euclidean')
        # ax.set_title('image15')
        # ax.imshow(pair_dist15.max()-pair_dist15)

        # ax = fig.add_subplot(153)
        # pair_dist30 = pw.pairwise_distances(image30,metric='euclidean')
        # ax.set_title('image30')
        # ax.imshow(pair_dist30.max()-pair_dist30)

        # ax = fig.add_subplot(154)
        # pair_dist60 = pw.pairwise_distances(image60,metric='euclidean')
        # ax.set_title('image60')
        # ax.imshow(pair_dist60.max()-pair_dist60)

        # ax = fig.add_subplot(155)
        # images = np.vstack(aligned_images)
        # ax.imshow(images)
        # plt.savefig('image.png')
        # plt.close()



        # shape_mean = np.mean(shape,axis=1)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(area,color,shape_mean)
        # ax.set_xlabel('X Label=area')
        # ax.set_ylabel('Y Label=color')
        # ax.set_zlabel('Z Label=shape')
        # plt.show()
            
        #####################################
    
        
            

        return component_features
    
    def concat_component_feature(self,component_features):
        """
        result: List[component1_feature,component2_feature,...]
        """
        result = dict()
        for feature in ['area','color','position','cnn_image']: #,'cnn_shape'
            result[feature] = list()
            for i in range(len(component_features)):
                result[feature].append(component_features[i][feature])
            result[feature] = np.array(result[feature])


        # for i in range(len(component_features)):
        #     result.append(np.concatenate([component_features[i][f] for f in feature_list],axis=0))
        return result
    
    def extract(self,image,masks):
        """
        image: [256,256,3] contains a component, value range [0,255]
        mask: [N,256,256] contains a component mask, value range [0,255]
        output: List[feature_dict]
        """
        component_features = self.compute_component_feature(image,masks)
        component_features = self.concat_component_feature(component_features)
        return component_features

    # def reduce_image_feature(self,component_features):
    #     """
    #         area_hist: [num_component]
    #         color_hist: [num_component]
    #         composite_hist: [num_component*feature_dim]
    #         position_hist: [num_component*2]
    #         shape_hist: [num_component*7]
    #     """
    #     area_hist = np.array([feature['area'] for feature in component_features])
    #     color_hist = np.array([feature['color'] for feature in component_features])
    #     composite_hist = np.concatenate([feature['cnn'] for feature in component_features],axis=0)
    #     position_hist = np.concatenate([feature['position'] for feature in component_features],axis=0)
    #     shape_hist = np.concatenate([feature['shape'] for feature in component_features],axis=0)
    #     print(f"area_hist:{area_hist.shape}")
    #     print(f"color_hist:{color_hist.shape}")
    #     print(f"composite_hist:{composite_hist.shape}")
    #     print(f"position_hist:{position_hist.shape}")
    #     print(f"shape_hist:{shape_hist.shape}")
    #     return area_hist,color_hist,composite_hist,position_hist,shape_hist
        


        
    # def reduce_feature(self,features):
    #     # features: [num_feature,dim] -> [global_feat_dim]
    #     if self.config['reduce_feature'] == 'mean':
    #         return torch.mean(features,dim=0)
    #     elif self.config['reduce_feature'] == 'histogram':
    #         return self.histogram(features)
    #     elif self.config['reduce_feature'] == 'kmeans':
    #         return self.histogram_by_kmeans(features,self.config['kmeans_cluster'])
    #     elif self.config['reduce_feature'] == 'mlp':
    #         return self.histogram_by_mlp(features,self.config['mlp'])

    
    # def histogram_by_kmeans(self,features,k_cluster):
    #     # features: [num_feature, dim] -> [k_cluster]
    #     # k_cluster: [k_cluster, dim]
    #     k_cluster.to(self.device)
    #     similarity_map = torch.matmul(features,k_cluster.T) # [num_feature,k]
    #     classified = torch.argmax(similarity_map,dim=1) # [num_feature]
    #     # build histogram
    #     histogram = torch.zeros(k_cluster.shape[0])
    #     for i in range(k_cluster.shape[0]):
    #         histogram[i] = (classified==i).sum()
    #     if self.config['normalize']:
    #         return F.normalize(histogram,p=2,dim=0)
    #     return histogram
        
    # def histogram_by_mlp(self,features,mlp):
    #     # This function only recives fixed number of features
    #     # features: [num_feature, dim] -> [mlp_dim]
    #     features = torch.flatten(features)
    #     with torch.no_grad():
    #         out = mlp(features)
    #     return out
        
    
    # def histogram(self,features):
    #     # features: [num_feature,dim] -> [bins*proj_dim]
    #     #histogram per channel
    #     if self.config['dim_reduction']:
    #         with torch.no_grad():
    #             features = self.projector(features)
    #     bins = self.config['bins']
    #     histogram = torch.zeros(features.shape[1],bins)
    #     for i in range(features.shape[1]):
    #         # filter out 5% and 95% quantile
    #         low = torch.quantile(features[:,i],0.05)
    #         high = torch.quantile(features[:,i],0.95)
    #         histogram[i] = torch.histc(features[:,i],bins=bins,min=low.item(),max=high.item())
    #     if self.config['normalize']:
    #         return torch.flatten(F.normalize(histogram,p=2,dim=0))
    #     return torch.flatten(histogram)