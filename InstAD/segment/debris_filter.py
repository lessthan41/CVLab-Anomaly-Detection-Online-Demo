import os
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from segment.instance_segment import load_backbone, remove_instances
from cnn.dataset import InstancesDataset
from sklearn.metrics import pairwise_distances

target_size = 256
batch_size = 32
# th = 0.295
th = 0.25
device = "cuda"
path = "/home/anomaly/data/segment/output/mpdd/"
classname = "tubes"
mode = "test"
# defect_types = ["good", "anomalous"]
defect_types = ["good"]
demo = False

for defect_type in defect_types:
	instance_path = os.path.join(path,"instance", classname, mode, defect_type)
	instance_mask_path = os.path.join(path,"instance_mask", classname, mode, defect_type)
	position_path = os.path.join(path,"position", classname, mode, defect_type)

	instances = [os.path.join(instance_path, i) for i in os.listdir(instance_path)]
	instance_masks = [os.path.join(instance_mask_path, i) for i in os.listdir(instance_mask_path)]
	positions = [os.path.join(position_path, i) for i in os.listdir(position_path)]

	backbone = load_backbone("resnet18", ["layer2","layer3"], imagesize=target_size, device=device)

	instance_dataset = InstancesDataset(
		imagesize=target_size, 
		path=instances, 
	)
	dataloader = torch.utils.data.DataLoader(
		instance_dataset,
		batch_size=batch_size,
		shuffle=False,
	)

	rm_list = []
	with tqdm.tqdm(dataloader, desc="Debris Filtering...", leave=False) as data_iterator:
		for i, batch in enumerate(data_iterator):
			image = batch["image"].to(device)
			inst_feature = backbone(image)
			inst_feature_numpy = inst_feature.cpu().numpy()
			dist = pairwise_distances(inst_feature_numpy, inst_feature_numpy, metric="l2")
			out_dist = [np.sum(dist[i]) for i in range(len(dist))]
			out_dist = out_dist / np.linalg.norm(out_dist) # normalize
			for j in range(len(out_dist)):
				if out_dist[j] > th:
					if demo:
						rm_list.append([out_dist[j], batch["image_path"][j]])
					else:
						rm_list.append(batch["image_path"][j])
	if not demo:
		remove_instances(rm_list)