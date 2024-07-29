import contextlib
import logging
import os
import gc
import numpy as np
import torch
import tqdm

from .patchcore import backbones
from .patchcore import common
from .patchcore import metrics
from .patchcore import patchcore
from .patchcore import sampler
from .patchcore import utils
from .patchcore import instance
from .patchcore import zeroshot

LOGGER = logging.getLogger(__name__)

_DATASETS = {"visa": ["InstAD.patchcore.datasets.visa", "VisADataset"],
             "demo": ["InstAD.patchcore.datasets.demo", "VisADataset"]}

class MyPatchcore():
    def __init__(self, inst_path, results_path, data_path, pos_path, class_name, batch_size, gpu, dist_metric, zero_shot, seed=0, demo=False) -> None:
        self.seed = seed
        self.inst_path = inst_path
        self.pos_path = pos_path
        self.class_name = class_name
        self.batch_size = batch_size
        self.demo = demo
        self.gpu = gpu
        self.result_collect = []
        self.list_of_dataloaders = []
        self.run_save_path = None
        self.device_context = None
        self.image_output_path = None
        self.zero_shot = zero_shot
        self.data_path = data_path
        self.results_path = results_path
        self.dist_metric = dist_metric
        self.device = utils.set_torch_device(self.gpu)
        # self.load_ckpt = False if self.zero_shot else load_ckpt
        self.methods = self.get_methods()
        self.patchcore_list = self.load()
        
    def get_methods(self):
        get_patchcore = self.patch_core()
        load_patchcore = self.patch_core_loader(patch_core_paths=[f"./InstAD/results/VisA_Results/fewshot/models/visa_{self.class_name}"])
        get_sampler = self.my_sampler()
        get_dataloaders = self.dataset()
        return [get_patchcore, load_patchcore, get_sampler, get_dataloaders]

    def load(
        self,
        log_group="fewshot",
        log_project="VisA_Results",
    ):
        methods = {key: item for (key, item) in self.methods}

        self.run_save_path = utils.create_storage_folder(
            self.results_path, log_project, log_group, mode="iterate"
        )

        self.list_of_dataloaders = methods["get_dataloaders"](self.seed, self.data_path, self.zero_shot)
        self.device_context = (
            torch.cuda.device("cuda:{}".format(self.device.index))
            if "cuda" in self.device.type.lower()
            else contextlib.suppress()
        )

        self.result_collect = []

        for dataloader_count, dataloaders in enumerate(self.list_of_dataloaders):
            if not self.demo:
                LOGGER.info(
                    "Evaluating dataset [{}] ({}/{})...".format(
                        dataloaders["training"].name,
                        dataloader_count + 1,
                        len(self.list_of_dataloaders),
                    )
                )

            utils.fix_seeds(self.seed, self.device)

            with self.device_context:
                torch.cuda.empty_cache()
                if self.demo and not self.zero_shot:
                    PatchCore_list = methods["get_patchcore_iter"]()
                    PatchCore = PatchCore_list[0]
                    LOGGER.info("Training model.")
                    torch.cuda.empty_cache()
                else:
                    imagesize = dataloaders["training"].dataset.imagesize
                    sampler = methods["get_sampler"](
                        self.device,
                    )
                    PatchCore_list, _, _ = methods["get_patchcore"](imagesize, sampler, self.device)
                    if len(PatchCore_list) > 1:
                        LOGGER.info(
                            "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                        )
                    for i, PatchCore in enumerate(PatchCore_list):
                        torch.cuda.empty_cache()
                        if PatchCore.backbone.seed is not None:
                            utils.fix_seeds(PatchCore.backbone.seed, self.device)
                        LOGGER.info(
                            "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                        )
                        torch.cuda.empty_cache()
                        if not self.zero_shot:
                            PatchCore.fit(dataloaders["training"])
        return PatchCore_list

    def run(
        self,
        image_path,
        output_image=True,
        save_patchcore_model=False,
    ):
        self.image_output_path = os.path.join(self.results_path, "output")
        if self.demo:
            dataset_info = _DATASETS["demo"]
            dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                image_path,
                self.inst_path,
                classname=[self.class_name],
                resize=256,
                data_path=self.data_path,
                zero_shot=self.zero_shot,
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            self.list_of_dataloaders[0]["testing"] = test_dataloader
        for dataloader_count, dataloaders in enumerate(self.list_of_dataloaders):
            with self.device_context:
                torch.cuda.empty_cache()
                aggregator = {"scores": [], "segmentations": []}
                for i, PatchCore in enumerate(self.patchcore_list):
                    torch.cuda.empty_cache()
                    feature_cluster = zeroshot.FeatureCluster(patchcore=PatchCore)
                    LOGGER.info(
                        "Embedding test data with models ({}/{})".format(
                            i + 1, len(self.patchcore_list)
                        )
                    )
                    anomaly_labels, masks_gt = [], []
                    if self.zero_shot:
                        nn_method = common.FaissNN(on_gpu=True, num_workers=4, metric=self.dist_metric)
                        data_to_iterate = dataloaders["testing"].dataset.data_to_iterate
                        data_to_iterate = [data_to_iterate[i:i+self.batch_size] for i in range(0, len(data_to_iterate), self.batch_size)]
                        imagesize = dataloaders["testing"].dataset.resize
                        with tqdm.tqdm(data_to_iterate, desc="Zero Shot Inferring...", leave=False) as data_iterator:
                            for data in data_iterator:
                                torch.cuda.empty_cache()
                                all_instances = []
                                for i in range(len(data)):
                                    classname, anomaly, instances, mask = data[i]
                                    all_instances += instances
                                features = feature_cluster.get_normal_features(all_instances)
                                ### redefine the patchcore.anomaly_scorer (knn with k=0.005 * len(features))
                                nn_method.reset_index()
                                PatchCore.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
                                    n_nearest_neighbours=int(np.min((2048,0.005*len(features)))), nn_method=nn_method, dist_metric=self.dist_metric
                                )
                                PatchCore.fit(features, is_feature=True)
                                test_instances = zeroshot.ImageDataset(
                                    imagesize=imagesize,
                                    instances=all_instances,
                                    split="test",
                                    data_path=self.data_path,
                                )
                                test_dataloader = torch.utils.data.DataLoader(
                                    test_instances,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                )
                                scores, segmentations, anomaly_label, mask_gt = PatchCore.predict(
                                    test_dataloader, self.data_path, os.path.join(self.pos_path, f"{os.path.basename(image_path)[:-4]}.pkl")
                                )
                                aggregator["scores"] += scores
                                aggregator["segmentations"] += segmentations
                                anomaly_labels += anomaly_label
                                masks_gt += mask_gt
                                PatchCore.anomaly_scorer = None
                                torch.cuda.empty_cache()
                        aggregator["scores"] = [aggregator["scores"]]
                        aggregator["segmentations"] = [aggregator["segmentations"]]
                    else:
                        scores, segmentations, anomaly_labels, masks_gt = PatchCore.predict(
                            dataloaders["testing"], self.data_path, os.path.join(self.pos_path, f"{os.path.basename(image_path)[:-4]}.pkl")
                        )

                        aggregator["scores"].append(scores)
                        aggregator["segmentations"].append(segmentations)

                # if not self.demo:
                scores = np.array(aggregator["scores"])
                # min_scores = scores.min(axis=-1).reshape(-1, 1)
                # max_scores = scores.max(axis=-1).reshape(-1, 1)
                # scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)

                segmentations = np.array(aggregator["segmentations"])
                # min_scores = (
                #     segmentations.reshape(len(segmentations), -1)
                #     .min(axis=-1)
                #     .reshape(-1, 1, 1, 1)
                # )
                # max_scores = (
                #     segmentations.reshape(len(segmentations), -1)
                #     .max(axis=-1)
                #     .reshape(-1, 1, 1, 1)
                # )
                # segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)

                ###################### modify here ###############################
                if output_image:
                    LOGGER.info("Output Image.")
                    image_paths = [
                        x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                    ]
                    if self.zero_shot:
                        image_paths = [x for y in image_paths for x in y]
                    utils.reset_folder(f"{self.image_output_path}/{self.class_name}_{self.dist_metric}")
                    if self.demo:
                        instance.output_img_demo(
                            scores, 
                            segmentations, 
                            masks_gt,
                            image_output_path=self.image_output_path, 
                            data_path=self.data_path, 
                            dist_metric=self.dist_metric,
                            class_name=self.class_name,
                            image_path=image_path,
                        )
                    else:
                        instance.output_img(
                            image_paths, 
                            scores, 
                            segmentations, 
                            masks_gt,
                            image_output_path=self.image_output_path, 
                            data_path=self.data_path, 
                            dist_metric=self.dist_metric,
                            class_name=self.class_name,
                        )
                ##################################################################

                if not self.demo:
                    LOGGER.info("Computing evaluation metrics.")
                    auroc = metrics.compute_imagewise_retrieval_metrics(
                        scores, anomaly_labels
                    )["auroc"]

                    # Compute PRO score & PW Auroc for all images
                    pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                        segmentations, masks_gt
                    )
                    full_pixel_auroc = pixel_scores["auroc"]
                    pixel_AP = pixel_scores["ap"]
                    fpr = pixel_scores["optimal_fpr"]
                    fnr = pixel_scores["optimal_fnr"]

                    # Compute PRO score & PW Auroc only images with anomalies
                    sel_idxs = []
                    for i in range(len(masks_gt)):
                        if np.sum(masks_gt[i]) > 0:
                            sel_idxs.append(i)
                    pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                        [segmentations[i] for i in sel_idxs],
                        [masks_gt[i] for i in sel_idxs],
                    )
                    anomaly_pixel_auroc = pixel_scores["auroc"]

                    self.result_collect.append(
                        {
                            "dataset_name": self.class_name,
                            "instance_auroc": auroc,
                            "full_pixel_auroc": full_pixel_auroc,
                            "anomaly_pixel_auroc": anomaly_pixel_auroc,
                            "pixel_AP": pixel_AP,
                            "fpr": fpr,
                            "fnr": fnr,
                        }
                    )

                    for key, item in self.result_collect[-1].items():
                        if key != "dataset_name":
                            LOGGER.info("{0}: {1:3.3f}".format(key, item))

                    # (Optional) Store PatchCore model for later re-use.
                    # SAVE all patchcores only if mean_threshold is passed?
                    if save_patchcore_model:
                        patchcore_save_path = os.path.join(
                            self.run_save_path, "models", self.class_name
                        )
                        os.makedirs(patchcore_save_path, exist_ok=True)
                        for i, PatchCore in enumerate(self.patchcore_list):
                            prepend = (
                                "Ensemble-{}-{}_".format(i + 1, len(self.patchcore_list))
                                if len(self.patchcore_list) > 1
                                else ""
                            )
                            PatchCore.save_to_path(patchcore_save_path, prepend)

            LOGGER.info("\n\n-----\n")

            # Store all results and mean scores to a csv-file.
            if not self.demo:
                result_metric_names = list(self.result_collect[-1].keys())[1:]
                result_dataset_names = [results["dataset_name"] for results in self.result_collect]
                result_scores = [list(results.values())[1:] for results in self.result_collect]
                utils.compute_and_store_final_results(
                    self.run_save_path,
                    result_scores,
                    column_names=result_metric_names,
                    row_names=result_dataset_names,
                )
        
        return scores[0], segmentations

    def patch_core(
        self,
        backbone_names=["wideresnet50"],
        layers_to_extract_from=("layer2","layer3"),
        pretrain_embed_dimension=1024,
        target_embed_dimension=512,
        patchsize=3,
        anomaly_scorer_num_nn=1,
        faiss_on_gpu=True,
        faiss_num_workers=8,
    ):
        backbone_names = list(backbone_names)
        if len(backbone_names) > 1:
            layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
            for layer in layers_to_extract_from:
                idx = int(layer.split(".")[0])
                layer = ".".join(layer.split(".")[1:])
                layers_to_extract_from_coll[idx].append(layer)
        else:
            layers_to_extract_from_coll = [layers_to_extract_from]

        def get_patchcore(input_shape, sampler, device):
            loaded_patchcores = []
            for backbone_name, layers_to_extract_from in zip(
                backbone_names, layers_to_extract_from_coll
            ):
                backbone_seed = None
                if ".seed-" in backbone_name:
                    backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                        backbone_name.split("-")[-1]
                    )
                backbone = backbones.load(backbone_name)
                backbone.name, backbone.seed = backbone_name, backbone_seed

                nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers, metric=self.dist_metric)

                patchcore_instance = patchcore.PatchCore(device)
                patchcore_instance.load(
                    backbone=backbone,
                    layers_to_extract_from=layers_to_extract_from,
                    device=device,
                    input_shape=input_shape,
                    pretrain_embed_dimension=pretrain_embed_dimension,
                    target_embed_dimension=target_embed_dimension,
                    patchsize=patchsize,
                    featuresampler=sampler,
                    anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                    nn_method=nn_method,
                )
                loaded_patchcores.append(patchcore_instance)
            return loaded_patchcores, target_embed_dimension, self.dist_metric

        return ("get_patchcore", get_patchcore)


    def my_sampler(name="identity", percentage=0.1):
        def get_sampler(device):
            if name == "identity":
                return sampler.IdentitySampler()
            elif name == "greedy_coreset":
                return sampler.GreedyCoresetSampler(percentage, device)
            elif name == "approx_greedy_coreset":
                return sampler.ApproximateGreedyCoresetSampler(percentage, device)

        return ("get_sampler", get_sampler)

    def dataset(
        self,
        train_val_split=1,
        k_shot=4,
        resize=256,
        num_workers=8,
        augment=False,
    ):
        name = "demo" if self.demo else "visa"
        dataset_info = _DATASETS[name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

        def get_dataloaders(seed, data_path, zero_shot=False):
            dataloaders = []
            for subdataset in [self.class_name]:
                if self.demo:
                    train_dataloader = None
                    val_dataloader = None
                    test_dataloader = None
                else:
                    train_dataset = dataset_library.__dict__[dataset_info[1]](
                        self.inst_path,
                        classname=subdataset,
                        resize=resize,
                        train_val_split=train_val_split,
                        split=dataset_library.DatasetSplit.TRAIN,
                        seed=seed,
                        augment=augment,
                        data_path=data_path,
                        zero_shot=zero_shot,
                        n_shot=k_shot,
                    )

                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    )

                    train_dataloader.name = name
                    if subdataset is not None:
                        train_dataloader.name += "_" + subdataset

                    if train_val_split < 1:
                        val_dataset = dataset_library.__dict__[dataset_info[1]](
                            self.inst_path,
                            classname=subdataset,
                            resize=resize,
                            train_val_split=train_val_split,
                            split=dataset_library.DatasetSplit.VAL,
                            seed=seed,
                            data_path=data_path,
                        )

                        val_dataloader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                        )
                    else:
                        val_dataloader = None

                    test_dataset = dataset_library.__dict__[dataset_info[1]](
                        self.inst_path,
                        classname=subdataset,
                        resize=resize,
                        split=dataset_library.DatasetSplit.TEST,
                        seed=seed,
                        data_path=data_path,
                        zero_shot=zero_shot,
                    )

                    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                    )

                dataloader_dict = {
                    "training": train_dataloader,
                    "validation": val_dataloader,
                    "testing": test_dataloader,
                }

                dataloaders.append(dataloader_dict)
            return dataloaders

        return ("get_dataloaders", get_dataloaders)

    def patch_core_loader(self, patch_core_paths="", faiss_on_gpu=True, faiss_num_workers=8):
        def get_patchcore_iter():
            loaded_patchcores = []
            for patch_core_path in patch_core_paths:
                gc.collect()
                n_patchcores = len(
                    [x for x in os.listdir(patch_core_path) if ".faiss" in x]
                )
                if n_patchcores == 1:
                    nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)
                    patchcore_instance = patchcore.PatchCore(self.device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path, device=self.device, nn_method=nn_method
                    )
                    loaded_patchcores.append(patchcore_instance)
                else:
                    for i in range(n_patchcores):
                        nn_method = common.FaissNN(
                            faiss_on_gpu, faiss_num_workers
                        )
                        patchcore_instance = patchcore.PatchCore(self.device)
                        patchcore_instance.load_from_path(
                            load_path=patch_core_path,
                            device=self.device,
                            nn_method=nn_method,
                            prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                        )
                        loaded_patchcores.append(patchcore_instance)

            return loaded_patchcores
        return ("get_patchcore_iter", get_patchcore_iter)