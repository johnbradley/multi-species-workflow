import argparse
from PytorchWildlife.models import detection
from PytorchWildlife import utils

def main():
    parser = argparse.ArgumentParser(description="Wildlife detection using MegaDetectorV6")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--input", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--annotated", type=str, required=True, help="Directory to annotated output images")
    parser.add_argument("--crop", type=str, required=True, help="Directory to cropped images")

    args = parser.parse_args()

    detection_model = detection.MegaDetectorV6(version=args.model) #"MDV6-yolov10-e")
    results = detection_model.batch_image_detection(data_path=args.input)#"./images/")
    utils.save_detection_images(results, args.annotated, overwrite=False)
    utils.save_crop_images(results, args.crop, overwrite=False)

if __name__ == "__main__":
    main()