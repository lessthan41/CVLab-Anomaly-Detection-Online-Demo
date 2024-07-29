import contextlib
import logging
import os
import sys
import click
import numpy as np
import torch
import tqdm

import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.metrics as metrics
import patchcore.patchcore as patchcore
import patchcore.sampler as sampler
import patchcore.utils as utils
import patchcore.instance as instance
import patchcore.zeroshot as zeroshot

LOGGER = logging.getLogger(__name__)

_DATASETS = {"visa": ["patchcore.datasets.visa", "VisADataset"],
             "mpdd": ["patchcore.datasets.visa", "VisADataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--image_output_path", required=True, type=str)
@click.option("--data_path", required=True, type=str)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--output_image", is_flag=True)
@click.option("--zero_shot", is_flag=True)
def main(**kwargs):
    pass

@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    image_output_path,
    data_path,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    output_image,
    zero_shot,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders, batch_size = methods["get_dataloaders"](seed, data_path, zero_shot)

    device = utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list, target_embed_dimension, dist_metric = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                if not zero_shot:
                    PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                feature_cluster = zeroshot.FeatureCluster(patchcore=PatchCore)
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                anomaly_labels, masks_gt = [], []
                pos_path = dataloaders["testing"].dataset.data_to_iterate[0][2].replace("instance","position").replace("_0.png",".pkl")
                if zero_shot:
                    nn_method = common.FaissNN(on_gpu=True, num_workers=4, metric=dist_metric)
                    data_to_iterate = dataloaders["testing"].dataset.data_to_iterate
                    data_to_iterate = [data_to_iterate[i:i+batch_size] for i in range(0, len(data_to_iterate), batch_size)]
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
                                n_nearest_neighbours=int(np.min((2048,0.005*len(features)))), nn_method=nn_method, dist_metric=dist_metric
                            )
                            PatchCore.fit(features, is_feature=True)
                            test_instances = zeroshot.ImageDataset(
                                imagesize=imagesize,
                                instances=all_instances,
                                split="test",
                                data_path=data_path,
                            )
                            test_dataloader = torch.utils.data.DataLoader(
                                test_instances,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                            )
                            scores, segmentations, anomaly_label, mask_gt = PatchCore.predict(
                                test_dataloader, data_path, pos_path
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
                        dataloaders["testing"], data_path, pos_path
                    )

                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            ###################### modify here ###############################
            if output_image:
                LOGGER.info("Output Image.")
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                if zero_shot:
                    image_paths = [x for y in image_paths for x in y]
                utils.reset_folder(f"{image_output_path}/{dataset_name.split('_')[1]}_{dist_metric}")
                instance.output_img(image_paths, scores, segmentations, masks_gt, 
                            epoch=-1, image_output_path=image_output_path, data_path=data_path, dist_metric=dist_metric)
            ##################################################################

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

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "pixel_AP": pixel_AP,
                    "fpr": fpr,
                    "fnr": fnr,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
## L1, L2, cosine, minkowski, mahalanobis
@click.option("--dist_metric", type=str, default="L1")
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    dist_metric,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers,
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

            nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers, metric=dist_metric)

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
        return loaded_patchcores, target_embed_dimension, dist_metric

    return ("get_patchcore", get_patchcore)


@main.command("my_sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def my_sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("inst_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--k_shot", default=4, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    inst_path,
    subdatasets,
    train_val_split,
    batch_size,
    k_shot,
    resize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, data_path, zero_shot=False):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                inst_path,
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

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                inst_path,
                classname=subdataset,
                resize=resize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
                data_path=data_path,
                zero_shot=zero_shot,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    inst_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                    data_path=data_path,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders, batch_size

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
