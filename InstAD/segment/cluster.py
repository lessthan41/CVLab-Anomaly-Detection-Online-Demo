# import os
# import argparse
# from segment.instance_segment import update_mean_feats, tsne
# from cnn.feature_extractor import PatchEmbedding
# from cnn.dataset import update_instances
# from utils import *

# def cluster(class_name, backbone, instance_dirs, img_id, pklpath, args):
#     ### get instances
#     instances = update_instances(img_id, instance_dirs)
#     mean_feats = update_mean_feats(backbone, img_id, instance_dirs, args.interval, args)
#     cluster_labels, cluster_centers = mean_shift(mean_feats)
#     tsne(mean_feats, instances, "/cluster/test/", "test.png", cluster_labels, output_path=f"/home/anomaly/research/segment/output/iter=-1")
#     instances = list(np.array(instances)[cluster_labels==0])
#     ### save instances
#     content = {"imgpaths_per_class":{"good":[]}, "data_to_iterate":[]}
#     if os.path.exists(pklpath):
#         content = load_pickle(pklpath)
#     content["imgpaths_per_class"]["good"] += instances
#     content["data_to_iterate"] += [[class_name, 'good', x, None] for x in instances]
#     save_pickle(pklpath, content)
#     return

# def find_cluster(args):
#     backbone = PatchEmbedding(backbone_name="resnet18", imagesize=args.target_size, layers_to_extract_from=["layer2","layer3"], device=args.device)
#     # backbone = YuShuanPatch(imagesize=args.target_size, device=args.device)
#     classes = os.listdir(args.inst_path)
#     for class_name in classes:
#         # if class_name != "capsules":
#         #     continue
#         print(class_name)
#         defect_path = os.path.join(args.data_path, class_name, "test")
#         pklpath = os.path.join(args.inst_path, class_name, "zero_shot.pkl")
#         os.remove(pklpath) if os.path.exists(pklpath) else None
        
#         ### get all images
#         all_imgs = []
#         defect_types = os.listdir(defect_path)
#         for defect in defect_types:
#             print(defect)
#             imgs = os.listdir(os.path.join(defect_path, defect))
#             instance_dir = os.path.join(args.inst_path,class_name,"test",defect)
#             all_imgs += [(os.path.splitext(i)[0], instance_dir) for i in imgs]
        
#         ### for image batch
#         inst_cnt = 0
#         img_id = []
#         instance_dirs = set()
#         for i, (img, instance_dir) in enumerate(tqdm.tqdm(all_imgs)):
#             img_id.append(img)
#             instance_dirs.add(instance_dir)
#             inst_cnt = get_instances_number(img_id, instance_dirs)
#             if inst_cnt >= args.cluster_batch_size or i==len(all_imgs)-1:
#                 cluster(
#                     class_name=class_name,
#                     backbone=backbone,
#                     instance_dirs=instance_dirs,
#                     img_id=img_id,
#                     pklpath=pklpath,
#                     args=args,
#                 )
#                 ### reset counter
#                 inst_cnt = 0
#                 img_id = []
#                 instance_dirs = set()

#     return

# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description="Description of your script.")
#     # parser.add_argument('-c', '--classes', type=str, help='classes seperated by ","', default='capsules')
#     parser.add_argument('-s', '--target_size', type=int, help='target size', default=256)
#     parser.add_argument('--inst_path', type=str, help='instance path', default='/mnt/sda/anomaly/segment/output/visa/instance/')
#     parser.add_argument('--data_path', type=str, help='data path', default='/mnt/sda/anomaly/VisA_highshot')
#     parser.add_argument('--inst_batch_size', type=int, help='instance batch size for feature extraction', default=1)
#     parser.add_argument('--cluster_batch_size', type=int, help='instance size for clustering', default=40)
#     parser.add_argument('--interval', type=int, help='interval', default=360)
#     parser.add_argument('--device', type=str, help='device', default='cuda')
#     args = parser.parse_args()

#     cluster = find_cluster(args)