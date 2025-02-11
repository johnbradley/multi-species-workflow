import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import supervision as sv
import argparse

# How to save masks

def parse_args():
    parser = argparse.ArgumentParser(description="Segment Anything")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image')
    parser.add_argument('--crop', type=str, help='Directory to save cropped images')
    parser.add_argument('--annotated', type=str, required=True, help='Directory to save the annotated images')
    parser.add_argument('--min_crop_size', type=str, required=True, help='Minimum crop width and height')

    return parser.parse_args()

args = parse_args()


sam_checkpoint = args.checkpoint #"sam_vit_h_4b8939.pth"
input_dir = args.input
output_dir = args.annotated
output_crop_path = args.crop
min_crop_size = int(args.min_crop_size)


# Load the model
model_type = "vit_h"  # Choose model type (vit_h, vit_l, vit_b)
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)


os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_crop_path, exist_ok=True)


mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

# Process each image in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_dir, filename)

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = mask_generator.generate(image_rgb)

        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_bgr.copy(), detections)
        annotated_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(annotated_image_path, annotated_image)

        for i in range(len(result)):
            x, y, width, height = result[i]['bbox']
            if width >= min_crop_size and height >= min_crop_size:
                cropped_image = image_bgr[int(y):int(y+height), int(x):int(x+width)]
                crop_path = os.path.join(output_crop_path, f"{i}_{filename}")
                cv2.imwrite(crop_path, cropped_image)
