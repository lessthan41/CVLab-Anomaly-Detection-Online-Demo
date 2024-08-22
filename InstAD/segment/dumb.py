import os
import glob

# count instances
for c in ["macaroni2"]:
# for c in ["macaroni1","macaroni2","candle"]:
	path = f"/home/anomaly/data/segment/output/visa/instance/{c}/test/"
	paths = os.listdir(path)
	paths = [os.path.join(path, i) for i in paths]

	for p in paths:
		_ = glob.glob(os.path.join(p, "*.*"))
		dct = {}
		for f in _:
			name = os.path.basename(f).split("_")[0]
			if name not in dct:
				dct[name] = 1
			else:
				dct[name] += 1

		for i in dct:
			if dct[i] != 4:
				print(c, i, dct[i], p)

# load pickle
# import pickle
# with open("/home/anomaly/data/segment/output/mpdd/position/tubes/train/good/006.pkl", 'rb') as f:
# 	a = pickle.load(f)
# print(a)
# print(len(a))

# tidy visa mask
# import glob
# path = "/home/anomaly/data/VisA_highshot"
# for classname in os.listdir(path):
# 	gt_path = os.path.join(path, classname, "ground_truth")
# 	for defect in os.listdir(gt_path):
# 		masks = glob.glob(os.path.join(gt_path, defect, "*.*"))
# 		for m in masks:
# 			front, back = os.path.split(m)
# 			if "_mask" not in back:
# 				id, ext = os.path.splitext(back)
# 				os.rename(m, os.path.join(front, id+"_mask"+ext))

# single image refinement
