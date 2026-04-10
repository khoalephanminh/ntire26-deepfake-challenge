import os
import random
import argparse
from copy import deepcopy

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
import zipfile
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Inference on custom test dataset.")
parser.add_argument(
    "--detector_path",
    type=str,
    default="abc.yaml",
    help="Path to detector YAML file.",
)
parser.add_argument("--test_dataset", nargs="+", help="Name(s) of test dataset as in dataset_json.")
parser.add_argument(
    "--weights_path",
    type=str,
    required=True,
    help="Path to trained weights checkpoint (ckpt_best.pth).",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="submission.txt",
    help="Path to output submission.txt.",
)
args = parser.parse_args()


def init_seed(config: dict) -> None:
    if config.get("manualSeed") is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config.get("cuda", False):
        torch.cuda.manual_seed_all(config["manualSeed"])


def prepare_testing_data(config: dict):
    def get_test_data_loader(config_local: dict, test_name: str):
        # work on a shallow copy so we don't mutate the shared config
        cfg = config_local.copy()
        cfg["test_dataset"] = test_name

        test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

        loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=cfg["test_batchSize"],
            shuffle=False,
            num_workers=int(cfg["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )
        return loader

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


@torch.no_grad()
def model_inference(model, data_dict: dict):
    return model(data_dict, inference=True)


def collect_predictions(model, data_loader):
    """Run the model and collect per-frame probabilities in loader order."""
    all_probs = []
    for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_dict["index"] = _
        # get data
        data, label, mask, landmark = (
            data_dict["image"],
            data_dict["label"],
            data_dict["mask"],
            data_dict["landmark"],
        )
        # binarize labels (0 vs 1) as in training/test.py
        label = torch.where(data_dict["label"] != 0, 1, 0)

        data_dict["image"], data_dict["label"] = data.to(device), label.to(device)
        if "image_2" in data_dict:
            data_dict["image_2"] = data_dict["image_2"].to(device)
        if "image_3" in data_dict:
            data_dict["image_3"] = data_dict["image_3"].to(device)
        if mask is not None:
            data_dict["mask"] = mask.to(device)
        if landmark is not None:
            data_dict["landmark"] = landmark.to(device)

        pred_dict = model_inference(model, data_dict)
        all_probs += list(pred_dict["prob"].cpu().detach().numpy())

    return np.array(all_probs)


def main():
    # 1) load configs
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open("./training/config/test_config.yaml", "r") as f:
        config_test = yaml.safe_load(f)
    config.update(config_test)

    # override datasets / weights from CLI
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    else:
        # ensure it's a list for our loop
        if isinstance(config["test_dataset"], str):
            config["test_dataset"] = [config["test_dataset"]]

    config["weights_path"] = args.weights_path

    # 2) seeds / cudnn
    init_seed(config)
    if config.get("cudnn", False):
        cudnn.benchmark = True

    # 3) data loaders
    test_data_loaders = prepare_testing_data(config)

    # 4) build detector and load weights
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    ckpt = torch.load(args.weights_path, map_location=device)
    # clean potential "module.backbone." prefix, mirroring training/test.py
    ckpt = {k.replace("module.backbone.", "backbone."): v for k, v in ckpt.items()}
    # drop obsolete CLIP buffers that don’t exist in the current model
    # ckpt = {k: v for k, v in ckpt.items() if "position_ids" not in k}

    # print("ckpt key:", ckpt.keys())
    # print("expected params:", model.state_dict().keys())
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 5) run inference and write submission file(s)
    # If multiple test datasets are requested, append the name to the base output_file stem.
    base_out = args.output_file
    base_root, base_ext = os.path.splitext(base_out)
    if os.path.exists(os.path.dirname(base_out)):
        shutil.rmtree(os.path.dirname(base_out))
    os.makedirs(os.path.dirname(base_out), exist_ok=True)

    for test_name, loader in test_data_loaders.items():
        probs = collect_predictions(model, loader)
        image_paths = loader.dataset.data_dict["image"]
        labels = loader.dataset.data_dict["label"]

        # sort by alphabetical order of filename
        pairs = list(zip(image_paths, probs, labels))
        pairs.sort(key=lambda x: os.path.basename(x[0]))

        if len(test_data_loaders) == 1:
            out_path = base_out
            out_path_full = base_out.replace(".txt", "_full.txt")
        else:
            out_path = f"{base_root}_{test_name}{base_ext or '.txt'}"
            out_path_full = f"{base_root}_{test_name}_full{base_ext or '.txt'}"

        with open(out_path, "w") as f:
            for _, p, l in pairs:  
                # one decimal place, still in [0.0, 1.0]
                f.write(f"{float(p):.1f}\n")

        with open(out_path_full, "w") as f_full:
            for _, p, l in pairs:  
                # Write to full precision file (unrounded)
                f_full.write(f"{float(p)}\n")
                
        print(f"[inference] Wrote {len(pairs)} predictions for {test_name} to {out_path}")

        # one output file -> zip it in its directory
        out_dir = os.path.dirname(out_path)
        zip_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=os.path.basename(out_path))
        print(f"[inference] Created zip: {zip_path}")


if __name__ == "__main__":
    main()
