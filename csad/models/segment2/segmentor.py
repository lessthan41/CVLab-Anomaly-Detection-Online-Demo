import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import tqdm
from collections import OrderedDict
import timm
from models.segment2.model import Segmentor, Segmentor2
from rich.progress import track
import os
from torchvision.ops import sigmoid_focal_loss
import sklearn

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        return self.ce_loss(probabilities,targets_one_hot)
    
class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        
        mod_a = intersection.sum()
        mod_b = targets.numel()
        
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss
    
class ClassBalancedDiceLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        probabilities = torch.softmax(prediction,dim=1)
        targets_one_hot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=prediction.shape[1])
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)


        class_weights = self._calculate_class_weights(targets_one_hot)
        dice_loss = self._dice_loss(probabilities, targets_one_hot)
        class_balanced_loss = class_weights * dice_loss
        return class_balanced_loss.mean()

    def _calculate_class_weights(self, target):
        """
        Calculates class weights based on their inverse frequency in the target.
        """
        weights = torch.zeros((target.shape[0],target.shape[1])).cuda()
        for c in range(target.shape[1]):
            weights[:,c] = 1 / (target[:,c].sum() + 1e-5)
        weights = weights / weights.sum(dim=1,keepdim=True)
        return weights.detach()

    def _dice_loss(self, prediction, target):
        """
        Calculates dice loss for each class and then averages across all classes.
        """
        intersection = 2 * (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5
        dice = (intersection + 1e-5) / (union + 1e-5)
        return 1 - dice
    

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, ignore_index=None, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.ignore_index = ignore_index

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        logit = torch.softmax(logit, dim=1)
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.ignore_index is not None:
            one_hot_key = torch.concat([one_hot_key[:,:self.ignore_index],one_hot_key[:,self.ignore_index+1:]],dim=1)
            logit = torch.concat([logit[:,:self.ignore_index],logit[:,self.ignore_index+1:]],dim=1)


        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss
    

class OldHistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits,targets):
        # ignore background
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(probabilities)

        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])

        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)
    
        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
    
        
        targets_hist = torch.mean(targets_one_hot,dim=(2,3)) # (B,C)
        targets_hist = torch.nn.functional.normalize(targets_hist,dim=1)
        
        preds_hist = torch.mean(probabilities,dim=(2,3)) # (B,C)
        preds_hist = torch.nn.functional.normalize(preds_hist,dim=1)

        hist_loss = torch.mean(torch.abs(targets_hist-preds_hist))
        return hist_loss
    
class HistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,pred, trg):
        pred = torch.softmax(pred,dim=1)
        new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
        new_trg = new_trg.scatter(1, trg, 1).float()
        diff = torch.abs(new_trg.mean((2, 3)) - pred.mean((2, 3)))
        if self.ignore_index is not None:
            diff = torch.concat([diff[:,:self.ignore_index],diff[:,self.ignore_index+1:]],dim=1)
        loss = diff.sum() / pred.shape[0]  # exclude BG
        return loss
    
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred):
        prob = nn.Softmax(dim=1)(pred)
        return (-1*prob*((prob+1e-5).log())).mean()
        
    

class SegmentorTrainer():
    def __init__(self,encoder,config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pad2resize_linear = self.config['pad2resize']
        self.encoder = encoder
        self.encoder.to(self.device)
        self.encoder.eval()
        #self.encoder.eval()
        
        self.image_size = config['image_size']
        self.num_classes = np.load(self.config['mask_root']+"/info/filtered_component_histogram.npy").shape[1] + 1
        os.makedirs(self.config['mask_root']+"/test_seg_output",exist_ok=True)
        self.segmentor = Segmentor(
            in_dim=self.config['in_dim'],
            num_classes=self.num_classes,
            in_size=self.image_size,
            pad2resize=self.pad2resize_linear
        )
        # self.segmentor = Segmentor2(
        #     in_dim=512,
        #     num_classes=self.num_classes,
        #     in_size=self.image_size,
        #     pad2resize=self.pad2resize_linear
        # )
        self.segmentor.to(self.device)

        self.color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
        
    def fit(self,sup_dataloader,unsup_dataloader,val_dataloaader,test_dataloader,full_train_dataloader):
        # training
        

        self.optimizer = torch.optim.Adam(self.segmentor.parameters(),lr=self.config["lr"],weight_decay=1e-5)
        # self.optimizer = torch.optim.SGD(self.segmentor.parameters(),lr=self.config["lr"],weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             T_max=self.config["epoch"],
        #                                                             eta_min=1e-4,
        #                                                             last_epoch=-1,
        #                                                             verbose=False)
        self.ce_loss = MulticlassCrossEntropyLoss(ignore_index=None)#nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(ignore_index=None)
        # self.dice_loss = MulticlassDiceLoss()
        self.dice_loss = ClassBalancedDiceLoss(ignore_index=None)
        self.hist_loss = HistLoss(ignore_index=0)
        self.entropy_loss = EntropyLoss()

        self.loss_dict = {
            "ce":self.ce_loss,
            "focal":self.focal_loss,
            "dice":self.dice_loss,
            "hist":self.hist_loss,
            "entropy":self.entropy_loss
        }
        self.loss_weight = self.config["loss_weight"]

        tqdm_obj = tqdm.tqdm(range(self.config["epoch"]),total=self.config["epoch"],desc="Training segmentor...")
        best_auc = 0
        best_logi_auc = 0
        best_strc_auc = 0

        iters_per_epoch = len(sup_dataloader)
        sup_dataloader_inf = InfiniteDataloader(sup_dataloader)
        unsup_dataloader_inf = InfiniteDataloader(unsup_dataloader)
        
        for epoch in tqdm_obj:
            sup_only = True if epoch < self.config["sup_only_epoch"] else False
            loss = self.train_one_epoch(tqdm_obj,sup_dataloader_inf,unsup_dataloader_inf,iters_per_epoch,sup_only)
            if (epoch+1)%5 == 0:
                self.test(test_dataloader)
                self.save(self.config["model_path"])
                logi_auc, strc_auc = self.test_hist_mahalanobis(self.segmentor,full_train_dataloader,val_dataloaader,test_dataloader,num_classes=self.num_classes)
                print(f"logical AUC: {logi_auc:.4f}| structural AUC: {strc_auc:.4f}")
                if (logi_auc+strc_auc)/2 > best_auc:
                    best_strc_auc = strc_auc
                    best_logi_auc = logi_auc
                    best_auc = (logi_auc+strc_auc)/2
                    self.test(test_dataloader)
                    self.validate(sup_dataloader,sup=True)
                    self.validate(unsup_dataloader,sup=False)
                    self.save(self.config["model_path"])
                    self.test(full_train_dataloader,save_train_image=True)
                    logi_auc, strc_auc = self.test_hist_mahalanobis(self.segmentor,full_train_dataloader,val_dataloaader,test_dataloader,num_classes=self.num_classes,save_score=True)
        
                
        
        logi_auc, strc_auc = self.test_hist_mahalanobis(self.segmentor,full_train_dataloader,val_dataloaader,test_dataloader,num_classes=self.num_classes)
        print(f"logical AUC: {logi_auc:.4f}| structural AUC: {strc_auc:.4f}")
        if (logi_auc+strc_auc)/2 > best_auc:
            best_auc = (logi_auc+strc_auc)/2
            best_logi_auc = logi_auc
            best_strc_auc = strc_auc
            self.test(test_dataloader)
            self.validate(sup_dataloader,sup=True)
            self.validate(unsup_dataloader,sup=False)
            self.save(self.config["model_path"])
            self.test(full_train_dataloader,save_train_image=True)
            logi_auc, strc_auc = self.test_hist_mahalanobis(self.segmentor,full_train_dataloader,val_dataloaader,test_dataloader,num_classes=self.num_classes,save_score=True)
        print(f"Best AUC: {best_auc:.4f}| Best logical AUC: {best_logi_auc:.4f}| Best structural AUC: {best_strc_auc:.4f}")
        # self.save(self.config["model_path"])

    def de_normalize(self,tensor):
        # tensor: (B,C,H,W)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return tensor[0]
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
            tensor = tensor * std.unsqueeze(0).unsqueeze(2).unsqueeze(3) + mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return tensor

    
    def train_one_epoch(self,tqdm_obj,sup_dataloader,unsup_dataloader,iters_per_epoch,sup_only=False):
        epoch_loss = 0
        for i,sup_batch,unsup_batch in zip(range(iters_per_epoch),sup_dataloader,unsup_dataloader):
            self.segmentor.train()

            image, gt, rand_gt, gt_path = sup_batch
            unsup_image, _, _, _ = unsup_batch

            with torch.no_grad():
                image = self.encoder(image)
                unsup_image = self.encoder(unsup_image)

            sup_out = self.segmentor(image)
            unsup_out = self.segmentor(unsup_image)
            
            sup_loss = 0
            sup_ce = self.loss_dict["ce"](sup_out,gt) * self.loss_weight['ce']
            sup_focal = self.loss_dict["focal"](sup_out,gt) * self.loss_weight['focal']
            sup_dice = self.loss_dict["dice"](sup_out,gt) * self.loss_weight['dice']
            sup_entro = self.loss_dict["entropy"](sup_out) * self.loss_weight['entropy']
            #sup_hist = self.loss_dict["hist"](sup_out,rand_gt) * self.loss_weight['hist']
            sup_loss = sup_ce + sup_focal + sup_dice #+ sup_hist

            unsup_loss = 0
            unsup_hist = self.loss_dict["hist"](unsup_out,rand_gt) * self.loss_weight['hist']
            unsup_entro = self.loss_dict["entropy"](unsup_out) * self.loss_weight['entropy']
            unsup_loss = unsup_hist + unsup_entro

            if sup_only:
                total_loss = sup_loss
            else:
                total_loss = sup_loss + unsup_loss + sup_entro
            epoch_loss += total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                tqdm_obj.set_description(
                    f"Current loss: {total_loss.item():.4f}| CE:{sup_ce.item():.4f}| Dice:{sup_dice.item():.4f}| Hist:{unsup_hist.item():.4f}| Focal:{sup_focal.item():.4f}| Entropy:{sup_entro.item():.4f}({unsup_entro.item():.4f}) ")
        # self.scheduler.step()
        return epoch_loss

    def save(self,path):
        torch.save(self.segmentor,path)
    
    def load(self,path):
        self.segmentor = torch.load(path)
        self.segmentor.to(self.device)

    def validate(self,dataloader,sup=False):
        self.segmentor.eval()
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                image, gt, rand_gt, gt_path = batch
                with torch.no_grad():
                    image_feat = self.encoder(image)
                out = self.segmentor(image_feat)
                for j in range(len(image)):
                    out_softmax = F.softmax(out[j],dim=0)
                    de_image = self.de_normalize(image[j])
                    de_image = de_image.squeeze(0).permute(1,2,0).cpu().numpy()
                    de_image = (de_image*255).astype(np.uint8)
                    de_image = cv2.cvtColor(de_image,cv2.COLOR_RGB2BGR)
                    out_softmax = torch.argmax(out_softmax,dim=0)
                    out_softmax = out_softmax.cpu().numpy()
                    color_out = np.zeros((self.image_size,self.image_size,3))
                    color_gt = np.zeros((self.image_size,self.image_size,3))
                    for k in range(1,self.num_classes):
                        color_out[out_softmax==k,:] = self.color_list[k-1]
                        color_gt[gt[j,0].cpu().numpy()==k,:] = self.color_list[k-1]
                    color_out = color_out.astype(np.uint8)
                    color_gt = color_gt.astype(np.uint8)
                    result = np.hstack([de_image,color_out,color_gt])
                    save_path = '/'.join(gt_path[j].replace("\\","/").split('/')[:-2])+"/val_seg_output"
                    image_name = gt_path[j].replace("\\","/").split("/")[-2]
                    os.makedirs(save_path,exist_ok=True)
                    # Image.fromarray(color_out).save(f'{gt_path[j].replace("filtered_cluster_map","pred_segmap_color")}.png')
                    # Image.fromarray(out_softmax.astype(np.uint8)).save(f'{gt_path[j].replace("filtered_cluster_map","pred_segmap")}.png')
                    # result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'{save_path}/{"sup" if sup else "unsup"}_{image_name}.png',result)
                    
        self.segmentor.train()

    def test(self,dataloader,save_train_image=False):
        self.segmentor.eval()
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                image,image_path = batch
                with torch.no_grad():
                    image_feat = self.encoder(image)
                out = self.segmentor(image_feat)

                out_softmax = F.softmax(out[0],dim=0)
                de_image = self.de_normalize(image[0])
                de_image = de_image.squeeze(0).permute(1,2,0).cpu().numpy()
                de_image = (de_image*255).astype(np.uint8)
                de_image = cv2.cvtColor(de_image,cv2.COLOR_RGB2BGR)
                out_softmax = torch.argmax(out_softmax,dim=0)
                out_softmax = out_softmax.cpu().numpy()
                color_out = np.zeros((self.image_size,self.image_size,3))
                color_gt = np.zeros((self.image_size,self.image_size,3))
                for k in range(1,self.num_classes):
                    color_out[out_softmax==k,:] = self.color_list[k-1]
                color_out = color_out.astype(np.uint8)
                save_path = image_path[0].replace("\\","/").replace("mvtec_loco_anomaly_detection","masks")
                anomaly_type = save_path.split("/")[-2]
                image_name = save_path.split("/")[-1].split(".")[0]
                save_dir = "/".join(os.path.dirname(save_path).split("/")[:-2])+"/test_seg_output"
                os.makedirs(save_dir,exist_ok=True)
                # Image.fromarray(color_out).save(f'{gt_path[j].replace("filtered_cluster_map","pred_segmap_color")}.png')
                if save_train_image:
                    Image.fromarray(out_softmax.astype(np.uint8)).save(f'{"/".join(os.path.dirname(save_path).split("/")[:-2])}/{image_name}/pred_segmap.png')
                else:
                    cv2.imwrite(f'{save_dir}/{anomaly_type}_{image_name}.png',color_out)

    def test_hist_mahalanobis(self,segmentor,train_loader,val_loader,test_loader,num_classes=3,save_score=False):
        num_classes = num_classes - 1
        segmentor.eval()
        def histogram(label_map,num_classes):
            hist = np.zeros(num_classes)
            for i in range(1,num_classes+1): # not include background
                hist[i-1] = (label_map == i).sum()
            hist = hist / label_map.size
            return hist
        true_score_logi = []
        pred_score_logi = []
        true_score_strc = []
        pred_score_strc = []
        segmentor.eval()
        # get train histograms
        train_hists = []
        with torch.no_grad():
            for image,path in train_loader:
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                train_hists.append(histogram(label_map,num_classes))
        train_hists = np.stack(train_hists,axis=0)
        mean = np.mean(train_hists,axis=0)
        from scipy.spatial.distance import mahalanobis
        def dist(x,data,mean):
            if data.shape[1] == 1:
                return np.linalg.norm(x-mean)
            else:
                cov = np.cov(data.T)
                return mahalanobis(x,mean,np.linalg.pinv(cov))
            
        val_scores = []
        for image,path in val_loader:
            with torch.no_grad():
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                hist = histogram(label_map,num_classes)
            score = dist(hist,train_hists,mean)
            val_scores.append(score)
        val_scores = np.array(val_scores)
        

        # get test histograms
        for image,path in test_loader:
            with torch.no_grad():
                with torch.no_grad():
                    image_feat = self.encoder(image)
                label_map = segmentor(image_feat)
                label_map = label_map.argmax(1)[0].cpu().numpy()
                hist = histogram(label_map,num_classes)
            score = dist(hist,train_hists,mean)
            
            

            save_path = path[0].replace("\\","/").replace("mvtec_loco_anomaly_detection","masks")
            anomaly_type = save_path.split("/")[-2]
            image_name = save_path.split("/")[-1].split(".")[0]
            if anomaly_type == "logical_anomalies":
                true_score_logi.append(1)
                pred_score_logi.append(score.item())
            elif anomaly_type == "structural_anomalies":
                true_score_strc.append(1)
                pred_score_strc.append(score.item())
            elif anomaly_type == "good":
                true_score_logi.append(0)
                true_score_strc.append(0)
                pred_score_logi.append(score.item())
                pred_score_strc.append(score.item())

        true_score_logi = np.array(true_score_logi)
        pred_score_logi = np.array(pred_score_logi)
        true_score_strc = np.array(true_score_strc)
        pred_score_strc = np.array(pred_score_strc)
        auc_logi = sklearn.metrics.roc_auc_score(true_score_logi,pred_score_logi)
        auc_strc = sklearn.metrics.roc_auc_score(true_score_strc,pred_score_strc)
        

        if save_score:
            category = save_path.split("/")[-4]
            np.save(f"./anomaly_score/{category}_hist_val_score.npy",val_scores)
            np.save(f"./anomaly_score/{category}_hist_logi_score.npy",pred_score_logi)
            np.save(f"./anomaly_score/{category}_hist_struc_score.npy",pred_score_strc)

        
        return auc_logi, auc_strc
    
if __name__ == "__main__":
    category = 'juice_bottle'
    config = {
        "image_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{category}/train/good/*.png",
        "mask_root":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/masks/{category}",
        "model_path":f"C:/Users/kev30/Desktop/anomaly/EfficientAD-res/ckpt/segmentor_{category}.pth ",
        "in_dim":[256,1024],
        "load":False,
        "image_size":512,
        "lr":1e-3,
        "epoch":150,
        "loss_weight":{
            "ce":1,
            "dice":1,
            "hist_entropy":1
        }
    }

    # dataset = SegmentDataset(image_path=config['image_path'],
    #                                   mask_root=config['mask_root'])
    # dataloader = DataLoader(dataset,batch_size=4,shuffle=False)
    encoder = timm.create_model('wide_resnet50_2.tv2_in1k'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[1,2,3])
    # encoder = timm.create_model('resnet18.tv_in1k'
    #                                       ,pretrained=True,
    #                                       features_only=True,
    #                                       out_indices=[1,2,3])
    # segmentor = Segmentor(encoder,in_dim=[64,256],num_classes=3).cuda()
    segmentor = Segmentor(encoder,in_dim=[256,1024],num_classes=3).cuda()
    segmentor.eval()
    a = torch.randn(1,3,256,256).cuda()
    out = segmentor(a)
    print(out.shape)




    # segmentor_trainer = SegmentorTrainer(encoder,config)
    # segmentor_trainer.eval()

    
    import time
    with torch.no_grad():
        times = []
        for i in range(2000):
            image = torch.randn(2,3,256,256,dtype=torch.float32).cuda()
            start = time.time()
            out = segmentor.predict(image)
            times.append(time.time()-start)
    print(np.mean(times[-100:]))

    print("Done!")