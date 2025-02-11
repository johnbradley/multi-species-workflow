import argparse
import os
from ultralytics import YOLO
import numpy as np
import cv2
import glob
import json


def main():
    parser = argparse.ArgumentParser(description="Parse an output directory and multiple files.")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("-m", "--model", required=True, help="Path to the YOLO model file")
    parser.add_argument("files", nargs='+', help="List of input image files")
    
    args = parser.parse_args()
    
    output_dir = args.output
    input_files = args.files
    model_name = args.model
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Output directory already exists: {output_dir}")
    
    # Load YOLO model
    model = YOLO(model_name)
    
    table_of_contents = []

    # Process each image
    for path in input_files:
        if os.path.isfile(path):
            print(f"Processing {path}...")
            results = model(path)
            image_filename = os.path.splitext(os.path.basename(path))[0]
            image_output_dir = os.path.join(output_dir, image_filename)
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            for i, result in enumerate(results):        
                # Save segmented image
                segmented_image_path = os.path.join(image_output_dir, f"segment_{i}.jpg")
                result.save(segmented_image_path)
                print(f"Segmented image saved: {segmented_image_path}")

                if result.masks is not None:
                    print("Saving masks...")
                    masks = result.masks.data.cpu().numpy()  # Get masks as numpy array
                    for j, mask in enumerate(masks):
                        mask_img = (mask * 255).astype(np.uint8)  # Convert mask to image format
                        mask_path = os.path.join(image_output_dir, f"mask_{i}_{j}.png")
                        cv2.imwrite(mask_path, mask_img)  # Save mask

                metadata_path = os.path.join(image_output_dir, f"result_{i}.json")
                with open(metadata_path, "w") as meta_file:
                    meta_file.write(str(result.to_json()))
                print(f"Metadata saved: {metadata_path}")

                print("Saving crops...")
                result.save_crop(image_output_dir, file_name=f"crop_{i}.png")

            image_name = os.path.basename(image_output_dir)
            print(f"Processing {image_name}...")
            crop_paths = glob.glob(f"{image_output_dir}/**/crop_*", recursive=True)
            mask_paths = glob.glob(f"{image_output_dir}/**/mask_*", recursive=True)
            result_paths = glob.glob(f"{image_output_dir}/**/result_*", recursive=True)
            segment_paths = glob.glob(f"{image_output_dir}/**/segment_*", recursive=True)
            details = {
                "image": image_name,
                "path": path,
                "crops": crop_paths,
                "masks": mask_paths,
                "results": result_paths,
                "segments": segment_paths,
            }
            print(details)
            table_of_contents.append(details)
        else:
            print(f"- {path} (Not found)")


    with open(os.path.join(output_dir, "table_of_contents.json"), "w") as toc_file:
        json.dump(table_of_contents, toc_file)

if __name__ == "__main__":
    main()
