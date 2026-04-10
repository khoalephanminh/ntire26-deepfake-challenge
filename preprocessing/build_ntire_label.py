import os
import json
import argparse

"""
python build_ntire_test_json.py \
    --image_dir /path/to/images \
    --output /path/to/output.json \
    --gt_path /raid/dtle/ntire26-deepfake-challenge/groudtruths/publictest_gt.txt
"""

def build_ntire_val_json(image_dir: str, output_path: str, dataset_name: str, gt_path: str) -> None:
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # Read ground truth labels into a list
    with open(gt_path, "r") as f:
        # Strip whitespace/newlines to get clean '0' or '1'
        gt_labels = [line.strip() for line in f.readlines() if line.strip() != ""]

    # Collect all PNG images and sort alphabetically
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    )
    if not filenames:
        raise RuntimeError(f"No .png images found in {image_dir}")

    # Initialize the data structure for 'real' (0) and 'fake' (1)
    data = {
        dataset_name: {
            "real": {
                "train": {},
                "val": {},
                "test": {}
            },
            "fake": {
                "train": {},
                "val": {},
                "test": {}
            }
        }
    }

    label_map = {"0": "real", "1": "fake"}
    cnt_write = 0

    for fname in filenames:
        if "_face" in fname:
            continue  # Skip any files that contain "_face" in their name
        
        video_name = os.path.splitext(fname)[0]  # e.g., "0000"
        
        try:
            # Convert "0000" to int 0 to index the ground truth list
            gt_index = int(video_name)
        except ValueError:
            print(f"Skipping {fname}: Could not parse an integer index from filename.")
            continue
            
        if gt_index >= len(gt_labels):
            raise IndexError(f"Image index {gt_index} ({fname}) is out of bounds for the GT file which has {len(gt_labels)} lines.")

        # Get '0' or '1' from the list, map it to 'real' or 'fake'
        gt_val = gt_labels[gt_index]
        actual_label = label_map.get(gt_val)
        
        if actual_label is None:
            print(f"Skipping {fname}: Unknown label '{gt_val}' at line {gt_index} in GT file.")
            continue

        frame_path = os.path.join(image_dir, fname)
        
        # Add to the correct category
        data[dataset_name][actual_label]["test"][video_name] = {
            "label": actual_label,
            "frames": [frame_path],
        }
        cnt_write += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(
        f"Wrote ntire_val JSON with {cnt_write} 1-frame videos "
        f"to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ntire_val.json where each image is a 1-frame video with actual labels."
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
    parser.add_argument(
        "--gt_path",
        default=(
            "/raid/dtle/ntire26-deepfake-challenge/groudtruths/publictest_gt.txt"
        ),
        help="Path to the ground truth text file (one label per line).",
    )
    args = parser.parse_args()

    # get datasetname from output
    datasetname = os.path.splitext(args.output)[0].split("/")[-1]

    build_ntire_val_json(args.image_dir, args.output, datasetname, args.gt_path)