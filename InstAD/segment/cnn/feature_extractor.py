import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
from . import common
from . import backbones
from .utils import PatchMaker

### config
warnings.filterwarnings("ignore")

class YuShuanPatch(nn.Module):
    def __init__(self, backbone_name="wide_resnet50_2", imagesize=256, device="cuda", normalize=True, mean=True):
        super(YuShuanPatch, self).__init__()
        self.device = device
        self.imagesize = imagesize
        self.mean = mean
        self.normalize = normalize
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(2,3))
        self.backbone.to(device)
        self.backbone.eval()
        self.avg = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        with torch.no_grad():
            x = self.backbone(x)
            out = []
            for i in range(len(x)):
                feat = x[i]
                feat = F.interpolate(feat,(32,32),mode='bilinear',align_corners=False)
                out.append(feat)
            out = torch.concat(out,dim=1)
            out = self.avg(out)
            ### normalize
            if self.normalize:
                out = F.normalize(out, dim=1)
            ### mean
            if self.mean:
                out = torch.mean(out,dim=(2,3))

            return out

class PatchEmbedding(nn.Module):
    def __init__(self, backbone_name="resnet18", imagesize=256, layers_to_extract_from=["layer2", "layer3"], device="cuda"):
        super(PatchEmbedding, self).__init__()
        self.device = device
        self.imagesize = imagesize
        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.patch_maker = PatchMaker(3, stride=1)

        self.backbone = backbones.load(self.backbone_name)
        self.backbone.to(device)
        self.backbone.eval()

        ### modules
        # backbone
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, device=self.device, train_backbone=False
        )
        self.forward_modules["feature_aggregator"] = feature_aggregator
        # preprocess
        self.feature_dimension = feature_aggregator.feature_dimensions((3,self.imagesize,self.imagesize))
        # print("feature_dimensions:", feature_dimensions)
        preprocessing = common.Preprocessing(
            input_dims=self.feature_dimension, output_dim=384
        )
        self.forward_modules["preprocessing"] = preprocessing
        # aggregator
        target_embed_dimension = 384
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator


    def forward(self, images, detach=True, provide_patch_shapes=False, evaluation=True):
        """Returns feature embeddings for images."""
        B = len(images)
        if not evaluation:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"].feature_extraction(images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"].feature_extraction(images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        # print("before patch features:", features[0].size(),
        #    "\n\tx: [bs, c, w, h]"
        # )
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        # print("after patch features:", features[0][0].size(),
        #    "\n\tx: [bs, w//stride*h//stride, c, patchsize, patchsize]"
        # )
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        
        features = [x.reshape(B, x.shape[-3], -1) for x in features]
        features = torch.cat(features, dim=1)
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim

        return features

class PatchEmbedding(nn.Module):
    def __init__(self, backbone_name="resnet18", imagesize=256, layers_to_extract_from=["layer2", "layer3"], device="cuda"):
        super(PatchEmbedding, self).__init__()
        self.device = device
        self.imagesize = imagesize
        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.patch_maker = PatchMaker(3, stride=1)

        self.backbone = backbones.load(self.backbone_name)
        self.backbone.to(device)
        self.backbone.eval()

        ### modules
        # backbone
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, device=self.device, train_backbone=False
        )
        self.forward_modules["feature_aggregator"] = feature_aggregator
        # preprocess
        self.feature_dimension = feature_aggregator.feature_dimensions((3,self.imagesize,self.imagesize))
        # print("feature_dimensions:", feature_dimensions)
        preprocessing = common.Preprocessing(
            input_dims=self.feature_dimension, output_dim=384
        )
        self.forward_modules["preprocessing"] = preprocessing
        # aggregator
        target_embed_dimension = 384
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator


    def forward(self, images, detach=True, provide_patch_shapes=False, evaluation=True):
        """Returns feature embeddings for images."""
        B = len(images)
        if not evaluation:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"].feature_extraction(images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"].feature_extraction(images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        # print("before patch features:", features[0].size(),
        #    "\n\tx: [bs, c, w, h]"
        # )
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        # print("after patch features:", features[0][0].size(),
        #    "\n\tx: [bs, w//stride*h//stride, c, patchsize, patchsize]"
        # )
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        
        features = [x.reshape(B, x.shape[-3], -1) for x in features]
        features = torch.cat(features, dim=1)

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        # features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        

        # batchsize x number_of_layers x input_dim -> batchsize x target_dim

        return features

if __name__ == "__main__":
    a = YuShuanPatch(imagesize=256,device="cuda")
    out = a(torch.randn(1,3,256,256).cuda())
    print(out.shape)