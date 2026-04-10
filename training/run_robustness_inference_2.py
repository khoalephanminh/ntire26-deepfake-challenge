# run_robustness_inference.py
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from dataset.customize_noise_augment import apply_targeted_degradation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Robustness Predictions (Batched)")
    parser.add_argument("--detector_path", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to tuned weights")
    parser.add_argument("--model_alias", type=str, required=True, help="e.g., 'crop', 'global', or 'fusion'")
    parser.add_argument("--test_dataset", type=str, default="publictest_data_final")
    parser.add_argument("--output_csv", type=str, default="robustness_predictions_002.csv")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    # Load Configs
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Safely load test_config.yaml
    test_config_path = "./training/config/test_config.yaml"
    if os.path.exists(test_config_path):
        with open(test_config_path, "r") as f:
            config_test = yaml.safe_load(f)
        config.update(config_test)
    
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset

    config["weights_path"] = args.weights_path

    # Define the experiment protocol
    protocols = {
        "jpeg": [100, 80, 60, 40, 20, 10, 5],
        "gaussian_noise": [0, 10, 20, 30, 40, 50, 60],
        "gaussian_blur": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "motion_blur": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "random_mix": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    }

    # Load Dataset
    print(f"Loading Dataset: {config['test_dataset']}...")
    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    
    batch_size = config.get("test_batchSize", 32)
    print(f"Using Batch Size: {batch_size}")

    # Load Model
    print(f"Loading Model: {args.model_alias}...")
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    
    ckpt = torch.load(args.weights_path, map_location=device)
    ckpt = {k.replace("module.backbone.", "backbone."): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    results = []
    total_images = len(dataset.data_dict['image'])

    print("Starting Batched Protocol Evaluation...")
    
    # Process in chunks
    for i in tqdm(range(0, total_images, batch_size)):
        end_idx = min(i + batch_size, total_images)
        
        # 1. Load clean images from disk JUST ONCE for this batch
        batch_img_np = []
        batch_labels = []
        batch_ids = []
        
        for idx in range(i, end_idx):
            image_path = dataset.data_dict['image'][idx]
            label = dataset.data_dict['label'][idx]
            
            img_pil = dataset.load_rgb(image_path)
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
            batch_img_np.append(img_np)
            batch_labels.append(int(label))
            batch_ids.append(os.path.basename(image_path))
            
        # 2. Iterate through all protocols for this specific batch
        for op_name, levels in protocols.items():
            for lvl in levels:
                
                batch_tensors = []
                
                # Apply noise on CPU (NumPy)
                for b_idx, img_np in enumerate(batch_img_np):
                    item_id = batch_ids[b_idx]
                    det_seed = hash(f"{item_id}_{op_name}_{lvl}") % (2**32)
                    
                    img_noisy = apply_targeted_degradation(img_np, op_name, lvl, seed=det_seed)
                    img_tensor = dataset.to_tensor(img_noisy)
                    img_norm = dataset.normalize(img_tensor)
                    batch_tensors.append(img_norm)
                    
                # Stack into a single GPU batch
                batch_tensor = torch.stack(batch_tensors).to(device)
                labels_tensor = torch.tensor(batch_labels).to(device)
                
                data_dict = {"image": batch_tensor, "label": labels_tensor}
                
                # Predict
                pred_dict = model(data_dict, inference=True)
                probs = pred_dict["prob"].cpu().numpy()
                
                # Save results
                for b_idx in range(len(batch_ids)):
                    results.append({
                        "model_alias": args.model_alias,
                        "item_id": batch_ids[b_idx],
                        "label": batch_labels[b_idx],
                        "noise_kind": op_name,
                        "noise_level": lvl,
                        "prediction": round(float(probs[b_idx]), 1)
                    })

    # Save/Append to CSV
    df_new = pd.DataFrame(results)
    if os.path.exists(args.output_csv):
        df_old = pd.read_csv(args.output_csv)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(args.output_csv, index=False)
    else:
        df_new.to_csv(args.output_csv, index=False)
        
    print(f"✅ Finished {args.model_alias}! Results appended to {args.output_csv}")

if __name__ == "__main__":
    main()