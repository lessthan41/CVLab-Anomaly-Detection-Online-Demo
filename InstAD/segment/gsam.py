import os
import cv2
import torch
import numpy as np
from PIL import Image

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_hf(filename, ckpt_config_filename, device='cpu'):
    # cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_config_file = ckpt_config_filename
    print(cache_config_file)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    # cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    cache_file = filename
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  

# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.5, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
    return annotated_frame, boxes 


def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def groundingdino(input_path, output_path, class_name, box_threshold, groundingdino_model, sam_predictor, is_invert=False, save_masks=True):
    
    # groundingdino_model = load_model_hf(CKPT_REPO_ID, CKPT_FILENAME, CKPT_CONFIG_FILENAME, device)
    # sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT).to(device))

    p = input_path
    # for p in tqdm.tqdm(input_path):
    ### record image size
    image_source, image = load_image(p)

    annotated_frame, detected_boxes = detect(image, image_source, text_prompt=f"background", box_threshold=box_threshold, model=groundingdino_model)
    Image.fromarray(annotated_frame)

    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
    annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
    Image.fromarray(annotated_frame_with_mask)

    # create mask images 
    mask = segmented_frame_masks[0][0].cpu().numpy()
    inverted_mask = ((1 - mask) * 255).astype(np.uint8)

    image_source_pil = Image.fromarray(image_source)
    image_mask_pil = Image.fromarray(mask)
    inverted_image_mask_pil = Image.fromarray(inverted_mask)
    
    if is_invert:
        if save_masks:
            image_mask_pil.save(f"{output_path}/{os.path.basename(p)}")
        return cv2.cvtColor(np.asarray(image_mask_pil), cv2.COLOR_RGB2BGR)
    else:
        if save_masks:
            inverted_image_mask_pil.save(f"{output_path}/{os.path.basename(p)}")
        return cv2.cvtColor(np.asarray(inverted_image_mask_pil), cv2.COLOR_RGB2BGR)
