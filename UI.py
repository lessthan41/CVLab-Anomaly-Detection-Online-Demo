import gradio as gr
import numpy as np
import PIL.Image as Image
import os
import glob
import time
import cv2
import random
from csad.main import CSAD
from InstAD.run_single_shot import InstAD as InstAD
from ShiZhi.main import main as ShiZhi

mvtec_ad_root = "/mnt/sda2/tokichan/MVTec-AD"
mvtec_loco_root = "/mnt/sda2/tokichan/MVTec_LOCO"
visa_root = "/mnt/sda2/tokichan/VisA_highshot"


def greet(model, dataset, class_name, input_image, index):
    image = Image.open(input_image[index][0]).convert("RGB")
    image_size = image.size
    global show_images

    if model == "CSAD":
        anomaly_map, anomaly_score = csad(image)
        # result_image = np.clip(anomaly_map, 0, 1)
        result_image = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        result_image = (result_image * 255).astype(np.uint8)
        gt_mask = Image.open(show_images[index].replace("test","ground_truth")).convert("RGB")
    elif model == "InstAD":
        anomaly_map, anomaly_score = instad(input_image[index][0])
        anomaly_map = anomaly_map[0]
        result_image = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        anomaly_score = np.max(result_image)
        result_image = (result_image * 200).astype(np.uint8)
        gt_mask = Image.open(show_images[index].replace("test","ground_truth").replace(".JPG","_mask.png")).convert("RGB")
    else:
        anomaly_map, anomaly_score = ShiZhi(class_name, image)

    result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_image)
    result_image = result_image.resize(image_size)

    # Define the text bar content
    text_bar = gr.Textbox(
        label=f"Anomaly Score:",
        value=f"{anomaly_score}",
        elem_id="anomaly_score",
    )
    
    return text_bar, result_image, gt_mask



def change_dataset(model):
    if model == "CSAD":
        return gr.update(choices=["MVTec LOCO"], value="MVTec LOCO")
    elif model == "InstAD":
        return gr.update(choices=["VisA"], value="VisA")
    else:
        return gr.update(choices=["MVTec AD"], value="MVTec AD")

def change_classes(dataset):
    if dataset == "MVTec LOCO":
        return gr.update(choices=["breakfast_box","pushpins","juice_bottle","screw_bag","splicing_connectors"], value="breakfast_box")
    elif dataset == "VisA":
        return gr.update(choices=["candle","capsules","macaroni1","macaroni2"], value="candle")
    else:
        return gr.update(choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"], value="bottle")

def show_class_images(dataset_name,class_name):
    num_images = 6
    global instad
    global csad
    if dataset_name == "MVTec LOCO":
        if "instad" in globals():
            del instad
        csad = CSAD(class_name)
        dataset_root = mvtec_loco_root
    elif dataset_name == "VisA":
        if "csad" in globals():
            del csad
        dataset_root = visa_root
        if "instad" in globals():
            instad.dataset_root = dataset_root
            instad.class_name = class_name
            instad.patchcore = instad.load_patchcore()
        else:
            instad = InstAD(dataset_root, class_name)
    else:
        dataset_root = mvtec_ad_root

    test_all_images = glob.glob(os.path.join(dataset_root, class_name, "test", "*/*.*"))
    test_good_images = glob.glob(os.path.join(dataset_root, class_name, "test", "good/*.*"))
    test_anomaly_images = [i for i in test_all_images if i not in test_good_images]

    global show_images
    show_images = random.sample(test_anomaly_images, num_images)
    return [Image.open(image_path) for image_path in show_images]

css="""
    #anomaly_score span {
        font-size: 25px;
    }
    #anomaly_score textarea {
        font-size: 25px;
    }
    h1 {
        font-size: 50px;
    }
    footer {
        display:none !important
    }
    #selected {
        visibility: hidden;
        width: 0px;
        height: 0px;
    }
"""

with gr.Blocks(css=css) as demo:
    text_bar = gr.Textbox(
        label="Anomaly Score:", 
        value="-",
        elem_id="anomaly_score",
    )
    
    model_switcher = gr.Dropdown(
        ["CSAD", "InstAD", "ShiZhi"], label="Model", info="Will not add more models later!"
    )
    dataset_switcher = gr.Dropdown(
        ["MVTec LOCO", "VisA", "MVTec AD"], label="Dataset", info="Will not add more datasets later!"
    )
    classes_switcher = gr.Dropdown(
        [], label="Dataset", info="Will not add more datasets later!"
    )

    class_test_gallery = gr.Gallery(
        label="Select image", show_label=False, elem_id="gallery"
    , columns=[3], rows=[2], object_fit="contain", height="auto")
    
    selected = gr.Number(show_label=False, elem_id="selected")

    def get_select_index(evt: gr.SelectData):
        return evt.index
    
    model_switcher.change(fn=change_dataset, inputs=model_switcher, outputs=dataset_switcher)
    dataset_switcher.change(fn=change_classes, inputs=dataset_switcher, outputs=classes_switcher)
    classes_switcher.change(fn=show_class_images, inputs=[dataset_switcher,classes_switcher], outputs=class_test_gallery)
    class_test_gallery.select(fn=get_select_index, outputs=selected)

    interface = gr.Interface(
        fn=greet,
        inputs=[model_switcher, dataset_switcher, classes_switcher, class_test_gallery, selected],  # Place text bar above the image input
        outputs=[text_bar, "image", "image"],
        title="CVLAB Anomaly Detection Online Demo",
    )

    demo.launch()
