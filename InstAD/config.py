amg_kwargs = {
    "points_per_side": 16,
    "points_per_batch": None,
    "pred_iou_thresh": None,
    "stability_score_thresh": 0.9,
    "stability_score_offset": None,
    "box_nms_thresh": 0.9,
    "crop_n_layers": 1,
    "crop_nms_thresh": 0.9,
    "crop_overlap_ratio": None,
    "crop_n_points_downscale_factor": None,
    "min_mask_region_area": None,
}

patchcore_kwargs = {
    'gpu': (0,), 
    'seed': 0, 
    'save_patchcore_model': True, 
    'image_output_path': '/mnt/sda2/tokichan/visualize/patchcore', 
    'data_path': '/mnt/sda2/tokichan/VisA_highshot', 
    'log_group': 'capsules', 
    'log_project': 'VisA_Results', 
    'results_path': 'results', 
    'save_segmentation_images': False, 
    'output_image': False, 
    'zero_shot': False
}