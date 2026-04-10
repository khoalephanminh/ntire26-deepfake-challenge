# filepath: /raid/dtle/thesis-Hoang-Khoa/NTIRE26-DeepfakeDetection/preprocessing/build_ntire_train_json.py
import os
import json
import argparse


def build_ntire_train_json(image_dir: str, output_path: str, dataset_name: str) -> None:
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect all PNG images and sort alphabetically
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png") or f.lower().endswith(".jpg")
    )
    if not filenames:
        raise RuntimeError(f"No .png or .jpg images found in {image_dir}")

    dataset_name = dataset_name

    # Two label groups: "real" and "fake"
    videos_real = {}
    videos_fake = {}
    cnt_write = 0

    for fname in filenames:
        if "_face" in fname:
            continue  # Skip any files that contain "_face" in their name
        cnt_write += 1

        base = os.path.splitext(fname)[0]  # e.g. "0000_real"
        parts = base.split("_")
        if len(parts) < 2:
            raise ValueError(f"Filename does not contain label part with '_': {fname}")
        if len(parts) > 2:
            continue  # skip files that don't match expected pattern

        label_tail = parts[-1].lower()  # "real" or "fake"
        if label_tail not in {"real", "fake"}:
            raise ValueError(f"Unsupported label in filename: {fname}")

        video_name = base  # unique video id, e.g. "0000_real"
        frame_path = os.path.join(image_dir, fname)

        entry = {
            "label": label_tail,     # "real" or "fake"
            "frames": [frame_path],  # 1-frame video
        }

        if label_tail == "real":
            videos_real[video_name] = entry
        else:
            videos_fake[video_name] = entry

    data = {
        dataset_name: {
            "real": {
                # duplicate data in both train and test
                "train": videos_real,
                "val": {},
                "test": videos_real,
            },
            "fake": {
                "train": videos_fake,
                "val": {},
                "test": videos_fake,
            },
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(
        f"Wrote ntire_train JSON with {cnt_write} 1-frame videos "
        f"to {output_path}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ntire_train.json where each image is a 1-frame video."
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

    build_ntire_train_json(args.image_dir, args.output, datasetname)
