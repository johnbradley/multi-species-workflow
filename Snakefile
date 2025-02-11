images_dir = "images"
metadata_file = "metadata.csv"
predict_species_path = "bioclip-predictions.csv"
yolo_segmented_dir = "yolo_segmented"
pywl_annotated_dir = "pywl_annotated"
pywl_crop_dir = "pywl_crop"
sega_annotated_dir = "sega_annotated"
sega_crop_dir = "sega_crop"

#TODO !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

rule segment_anything:
    conda:
        "envs/segment-anything.yaml"
    input:
        images_dir,
    output:
        annotated=directory(sega_annotated_dir),
        crop=directory(sega_crop_dir),
    shell:
        "python scripts/segment-anything.py --min_crop_size=224 --checkpoint sam_vit_h_4b8939.pth --input {input} --annotated {output.annotated} --crop {output.crop}"


rule pytorch_wildlife_segment_many:
    conda:
        "envs/pytorch-wildlife.yaml"
    input:
        images_dir,
    output:
        annotated=directory(pywl_annotated_dir),
        crop=directory(pywl_crop_dir),
    shell:
        "python scripts/pytorch-wildlife-segment.py --model MDV6-yolov10-e --input {input} --annotated {output.annotated} --crop {output.crop}"


rule predict_species:
    conda:
        "envs/predict-species.yaml"
    input:
        images=images_dir,
        yolo_segmented_dir=yolo_segmented_dir,
    output:
        predict_species_path
    shell:
        "bioclip predict --k 1 {input.images}/* {yolo_segmented_dir}/*/*/crop* > {output}"


rule yolo_segment_many:
    conda:
        "envs/yolo-segment.yaml"
    input:
        images_dir,
    output:
        directory(yolo_segmented_dir)
    shell:
        "python scripts/yolo-segment.py --model yolo11l-seg.pt --output {output} {input}/*"


rule download:
    conda:
        "envs/download.yaml"
    params:
        dataset="johnbradley/multi-species-images"
    output:
        images=directory(images_dir),
        metadata=metadata_file
    shell:
        "python scripts/download.py --dataset {params.dataset} --dsname train --images {output.images} --metadata {output.metadata}"
