
"""
CUDA_VISIBLE_DEVICES=7 python training/extract_cvpr_metrics.py \
    --detector_path "training/config/detector/dinov2_clip_ft_phase2.yaml" \
    --weights_path "pretrained_weights/dinov2_clip.pth" \
    --test_dataset "publictest_data_final"
"""

import os
import random
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as ss

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.noise_augment_wrapper import NoiseAugmentWrapper
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CVPR Paper Metrics (Cosine Sim & Attention Entropy)")
    parser.add_argument("--detector_path", type=str, required=True, help="Path to your yaml config (e.g., dinov2_252.yaml)")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to your tuned weights")
    parser.add_argument("--test_dataset", type=str, default="ntire_val", help="Dataset to extract from")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of real and fake samples each (1000 total)")
    return parser.parse_args()

def init_seed(config: dict) -> None:
    if config.get("manualSeed") is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config.get("cuda", False):
        torch.cuda.manual_seed_all(config["manualSeed"])

def calculate_attention_entropy(attentions):
    """
    attentions: Tensor of shape [1, num_heads, seq_len, seq_len] from the final ViT layer.
    We extract the CLS token's attention to all spatial patches, average across heads, and calculate entropy.
    """
    cls_attention = attentions[0, :, 0, 1:] 
    mean_attention = cls_attention.mean(dim=0).cpu().numpy()
    entropy = ss.entropy(mean_attention)
    return entropy

@torch.no_grad()
def main():
    args = parse_args()

    # 1. Load Configs (MATCHING INFERENCE.PY)
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

    # 2. Seeds / CuDNN
    init_seed(config)
    if config.get("cudnn", False):
        cudnn.benchmark = True

    # 3. Load Dataset
    print(f"Loading Dataset: {config['test_dataset']}...")
    # Work on a shallow copy as in prepare_testing_data
    cfg = config.copy()
    if isinstance(cfg["test_dataset"], list):
        cfg["test_dataset"] = cfg["test_dataset"][0]
        
    dataset = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

    # 4. Smart Sampling (Handles Labeled vs. Unlabeled datasets)
    real_indices = [i for i, label in enumerate(dataset.data_dict['label']) if int(label) == 0]
    fake_indices = [i for i, label in enumerate(dataset.data_dict['label']) if int(label) == 1]
    
    if len(real_indices) > 0 and len(fake_indices) > 0:
        print(f"Found Labels! Sampling {args.num_samples} Real and {args.num_samples} Fake...")
        selected_indices = random.sample(real_indices, min(args.num_samples, len(real_indices))) + \
                           random.sample(fake_indices, min(args.num_samples, len(fake_indices)))
    else:
        total_images = len(dataset.data_dict['image'])
        sample_size = min(args.num_samples * 2, total_images)
        print(f"No valid class split found. Processing {sample_size} images sequentially...")
        
        # THE FIX: Just take the first N indices in perfectly sorted, deterministic order
        selected_indices = list(range(sample_size))

    # 5. Build Detector and Load Weights (MATCHING INFERENCE.PY)
    print("Loading Model & Weights...")
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    
    ckpt = torch.load(args.weights_path, map_location=device)
    ckpt = {k.replace("module.backbone.", "backbone."): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 6. Initialize Noise Wrapper
    cfg_detector = config.get("degradation", {})
    degradation_config = {
        **cfg_detector,
        "type": cfg_detector.get("type", "ours"),
        "global_p": cfg_detector.get("global_p", 0.8),
        "val_global_p": cfg_detector.get("val_global_p", 0.0),
        "op_p": cfg_detector.get("op_p", 0.5),
        "degradations_strength": cfg_detector.get("degradations_strength", 0.8), # We will override this with specific noise levels during extraction
        "use_beta": False, # Disable random distribution for controlled noise levels
        "distractor_p": cfg_detector.get("distractor_p", 0.15),
        "a": cfg_detector.get("a", 1.2),
        "b": cfg_detector.get("b", 1.2),
        "mean": config.get("mean", [0.485, 0.456, 0.406]), 
        "std":  config.get("std",  [0.229, 0.224, 0.225]),
    }

    wrapper = NoiseAugmentWrapper(dataset, degradation_config, split="test")
    wrapper.op_p = 1.0  # Force degradation operations to trigger
    wrapper.use_beta = False # Disable random distribution
    wrapper.distractor_p = 0.0 # Disable distractions for pure noise evaluation

    # 7. Define Noise Thresholds to Test
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # You can adjust these levels as needed
    results = []

    print("Starting Metric Extraction...")
    for idx in tqdm(selected_indices, total=len(selected_indices)):
        image_path = dataset.data_dict['image'][idx]
        label = dataset.data_dict['label'][idx]
        
        # Load Raw Image
        img_pil = dataset.load_rgb(image_path)
        img_np_base = np.array(img_pil).astype(np.float32) / 255.0

        clean_cls_embedding = None

        for noise in noise_levels:
            # Apply strict noise level
            if noise == 0.0:
                img_noisy = img_np_base.copy()
            else:
                wrapper.strength = noise  # Force specific strength
                img_noisy = wrapper.degradation_fn(img_np_base.copy())

            # Convert to tensor and normalize exactly as dataset.__getitem__ would
            img_tensor = dataset.to_tensor(img_noisy)
            img_norm = dataset.normalize(img_tensor).unsqueeze(0).to(device)

            # FORWARD PASS (Extracting features and attentions)
            # --- THE FIX: Bypass PEFT Wrapper and Force Eager Attention ---
            # 1. Extract the raw Hugging Face model (LoRA weights are already injected in-place)
            core_hf_model = model.get_backbone().vision_model
            if hasattr(core_hf_model, "base_model"):
                core_hf_model = core_hf_model.base_model
                if hasattr(core_hf_model, "model"):
                    core_hf_model = core_hf_model.model
            
            # 2. Disarm the SDPA (Flash Attention) lock inside Hugging Face
            if hasattr(core_hf_model.config, "_attn_implementation"):
                core_hf_model.config._attn_implementation = "eager"
            
            core_hf_model.config.output_attentions = True

            # 3. Forward Pass
            out = core_hf_model(pixel_values=img_norm, output_attentions=True, return_dict=True)
            
            # 4. Safely route outputs
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                current_cls = out.pooler_output
            else:
                current_cls = out.last_hidden_state[:, 0]
                
            if out.attentions is None or len(out.attentions) == 0:
                raise ValueError("Hugging Face completely refused to output attentions. Check your Transformers version.")
            else:
                attentions = out.attentions[-1] # Get final layer attention map
            
            # Compute Metrics
            entropy = calculate_attention_entropy(attentions)
            # ----------------------------------------------------------------
            
            # Compute Metrics
            entropy = calculate_attention_entropy(attentions)
            
            if noise == 0.0:
                clean_cls_embedding = current_cls.clone()
                cosine_sim = 1.0 # Clean vs Clean is exactly 1.0
            else:
                cosine_sim = F.cosine_similarity(clean_cls_embedding, current_cls, dim=-1).item()

            results.append({
                "image": os.path.basename(image_path),
                "label": "Real" if int(label) == 0 else "Fake" if int(label) == 1 else "Unknown",
                "noise_level": noise,
                "attention_entropy": entropy,
                "cosine_similarity": cosine_sim
            })

    # Save to CSV
    df = pd.DataFrame(results)
    save_dir = "extracted_metrics"
    os.makedirs(save_dir, exist_ok=True)
    # output_csv = f"{save_dir}/cvpr_metrics_{cfg['backbone_name']}_{cfg['test_dataset']}.csv"
    output_csv = f"{save_dir}/cvpr_metrics_{cfg['backbone_name']}_noaug_{cfg['test_dataset']}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Extraction Complete! Saved to {output_csv}")

if __name__ == "__main__":
    main()