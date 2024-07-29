import os
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from .utils import save_pickle, color_masks

def intersect_ratio(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    if intersection.sum() == 0:
        return 0
    # ratio = np.sum(intersection)/min([np.sum(mask1!=0),np.sum(mask2!=0)])
    ratio = np.sum(intersection)/np.sum(mask1!=0)
    ratio = 0 if np.isnan(ratio) else ratio
    return ratio

### segment anything
### https://github.com/facebookresearch/segment-anything
def segment_anything(sam, input, output, model_type, device, amg_kwargs, save_masks=True, log=True):
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", **amg_kwargs)

    if not os.path.isdir(input):
        targets = [input]
    else:
        targets = [
            f for f in os.listdir(input) if not os.path.isdir(os.path.join(input, f))
        ]
        targets = [os.path.join(input, f) for f in targets]

    for t in targets:
        print(f"Processing '{t}'...") if log else None
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_size = (image.shape[1], image.shape[0])
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        masks = generator.generate(image)
        base = os.path.basename(t)
        filename = os.path.join(output, base) if save_masks else None

        ### aggregate all masks into single mask in masks
        _mask = []
        for mask in masks:
            _mask.append(mask["segmentation"])
        
        remove_idx = []
        for i in range(len(_mask)):
            for j in range(i+1,len(_mask)):
                intersect_ratio_ij = intersect_ratio(_mask[i],_mask[j])
                intersect_ratio_ji = intersect_ratio(_mask[j],_mask[i])
                if intersect_ratio_ij > 0.9:
                    if intersect_ratio_ji > 0.5:
                        remove_idx.append(j)
                elif intersect_ratio_ji > 0.9:
                    if intersect_ratio_ij > 0.5:
                        remove_idx.append(i)
        
        remove_idx = list(set(remove_idx))
        _mask = [mask for i,mask in enumerate(_mask) if i not in remove_idx]
        # _mask = [cv2.resize(mask.astype(np.uint8)*255, image_size, interpolation=cv2.INTER_LINEAR) for mask in _mask]
        # _mask = [mask.astype(np.bool) for mask in _mask]
        color_mask = color_masks(_mask)
        if save_masks:
            save_pickle(filename[:-4]+".pkl", _mask)
            cv2.imwrite(filename, color_mask)
        
        return color_mask, _mask
