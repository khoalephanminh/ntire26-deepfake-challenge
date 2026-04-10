import os
import json
import argparse

"""
python build_ntire_test_json.py --image_dir /path/to/images --output /path/to/output.json
"""

def build_ntire_val_json(image_dir: str, output_path: str, dataset_name: str) -> None:
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect all PNG images and sort alphabetically
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    )
    if not filenames:
        raise RuntimeError(f"No .png images found in {image_dir}")

    dataset_name = dataset_name
    label_name = "ntire_val_dummy"  # must match entry in training/config/test_config.yaml

    # Each image becomes a separate "video" with a single frame
    videos = {}
    cnt_write = 0
    for fname in filenames:
        if "_face" in fname:
            continue  # Skip any files that contain "_face" in their name
        cnt_write += 1
        video_name = os.path.splitext(fname)[0]  # e.g., "0000"
        frame_path = os.path.join(image_dir, fname)
        videos[video_name] = {
            "label": label_name,
            "frames": [frame_path],
        }

    data = {
        dataset_name: {
            label_name: {
                "train": {},
                "val": {},
                "test": videos,
            }
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(
        f"Wrote ntire_val JSON with {cnt_write} 1-frame videos "
        f"to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ntire_val.json where each image is a 1-frame video."
    )
    parser.add_argument(
        "--image_dir",
        default=(
            "/raid/dtle/NTIRE26-DeepfakeDetection/"
            "datasets/unpreprocessed/publictest_data_final_cropped_original"
        ),
        help="Folder containing 0000.png, 0001.png, ...",
    )
    parser.add_argument(
        "--output",
        default=(
            "/raid/dtle/NTIRE26-DeepfakeDetection/"
            "preprocessing/dataset_json/ntire_test_cropped_original.json"
        ),
        help="Path to output JSON file.",
    )
    args = parser.parse_args()

    # get datasetname from output
    datasetname = os.path.splitext(args.output)[0].split("/")[-1]

    build_ntire_val_json(args.image_dir, args.output, datasetname)

