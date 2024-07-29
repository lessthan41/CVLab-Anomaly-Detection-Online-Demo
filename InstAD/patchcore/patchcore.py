"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from . import backbones
from . import common
from . import sampler
from . import instance

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        gaussian = common.GaussianBlur()
        self.forward_modules["gaussian"] = gaussian

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_scorer_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
            # features = self.forward_modules["gaussian"](features)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
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
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    
    def clear_memory_bank(self):
        self.anomaly_scorer.clear_memory_bank()

    def fit(self, training_data, is_feature=False):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, is_feature)

    def _fill_memory_bank(self, input_data, is_feature):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        if is_feature:
            features = input_data
        else:
            features = []
            with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    features.append(_image_to_features(image))

            features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data, data_path, pos_path):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, data_path, pos_path)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, data_path, pos_path):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        ### binding output ###
        scores_bd = []
        masks_bd = []
        labels_gt_bd = []
        masks_gt_bd = []
        img_paths_bd = []
        scores_output = []
        masks_output = []
        labels_gt_output = []
        masks_gt_output = []
        img_paths_output = []
        prev_img_id = None
        ######################

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

                #############################
                # binding output
                batch_size = len(_scores)
                for i in range(batch_size): # for i in batch
                    if prev_img_id is None:
                        prev_img_id = os.path.basename(data['image_path'][0]).split('_')[0]
                        scores_bd.append(_scores[0])
                        masks_bd.append(_masks[0])
                        labels_gt_bd.append(labels_gt[0])
                        masks_gt_bd.append(masks_gt[0])
                        img_paths_bd.append(img_paths[0])
                    else:
                        img_id = os.path.basename(data['image_path'][i]).split('_')[0]
                        if img_id != prev_img_id: # output final anomaly map and reset buffer
                            scores_new, segmentations_new, labels_gt_new, masks_gt_new = \
                            instance.generate_final_anomaly_maps(
                                scores_bd, masks_bd, labels_gt_bd, masks_gt_bd, img_paths_bd, data_path, dataloader.dataset.classnames_to_use[0], pos_path
                            )
                            ### append to final output ###
                            scores_output.extend(scores_new)
                            masks_output.extend(segmentations_new) # (329,329)
                            labels_gt_output.extend(labels_gt_new)
                            masks_gt_output.extend(masks_gt_new)
                            img_paths_output.extend(img_paths_bd)
                            ### reset buffer ###
                            img_paths = img_paths[(-1*batch_size):]
                            scores = scores[(-1*batch_size):]
                            masks = masks[(-1*batch_size):]
                            labels_gt = labels_gt[(-1*batch_size):]
                            masks_gt = masks_gt[(-1*batch_size):]
                            scores_bd = []
                            masks_bd = []
                            labels_gt_bd = []
                            masks_gt_bd = []
                            img_paths_bd = []
                            ### update prev_img_id ###
                            prev_img_id = img_id

                        scores_bd.append(_scores[i])
                        masks_bd.append(_masks[i])
                        labels_gt_bd.append(labels_gt[i-batch_size])
                        masks_gt_bd.append(masks_gt[i-batch_size])
                        img_paths_bd.append(img_paths[i-batch_size])

            scores_new, segmentations_new, labels_gt_new, masks_gt_new = \
            instance.generate_final_anomaly_maps(
                scores_bd, masks_bd, labels_gt_bd, masks_gt_bd, img_paths_bd, data_path, dataloader.dataset.classnames_to_use[0], pos_path
            )
            
            ### append to final output ###
            scores_output.extend(scores_new)
            masks_output.extend(segmentations_new)
            labels_gt_output.extend(labels_gt_new)
            masks_gt_output.extend(masks_gt_new)
            img_paths_output.extend(img_paths_bd)
            #############################

        return scores_output, masks_output, labels_gt_output, masks_gt_output

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: common.FaissNN,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
