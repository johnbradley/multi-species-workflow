import os
import argparse
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download and process a HuggingFace dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the HuggingFace dataset')
    parser.add_argument('--dsname', type=str, required=True, help='Name of the dataset to download')
    parser.add_argument('--images', type=str, required=True, help='Directory to save images')
    parser.add_argument('--metadata', type=str, required=True, help='Path to save metadata')
    return parser.parse_args()

def main(dataset, dsname, metadata, images_dir):
    if not os.path.exists(images_dir):
        print(f"Creating directory {images_dir}")
        os.makedirs(images_dir)

    print(f"Fetching dataset {dataset}...")
    ds = load_dataset(dataset)

    print(f"Downloading images to {images_dir}")
    paths = []
    for row in ds[dsname]:
        ext = row['url'].split('.')[-1]
        filename = f"{row['id']}.{ext}"
        dest = os.path.join(images_dir, filename)
        paths.append(dest)
        print(f"Downloading {dest}")
        row['image'].save(dest)

    print(f"Saving metadata to '{metadata}'.")
    df = ds[dsname].to_pandas()
    df = df.drop(columns=['image'])    
    df['path'] = paths
    df.to_csv(metadata, index=False)

    print("Done")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset, args.dsname, args.metadata, args.images)
