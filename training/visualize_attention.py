import os
import random
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import torch
import torch.backends.cudnn as cudnn

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.noise_augment_wrapper import NoiseAugmentWrapper
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    # --- NEW: Argparse integration ---
    parser = argparse.ArgumentParser(description="Visualize Attention Collapse under Degradation")
    parser.add_argument('--detector_path', type=str, default="training/config/detector/dinov2_252.yaml", help="Path to model config")
    parser.add_argument('--weights_path', type=str, default="pretrained_weights/dinov2_252_crop.pth", help="Path to .pth weights")
    parser.add_argument('--test_dataset', type=str, default="publictest_data_final_cropped_original", help="Dataset name in config")
    parser.add_argument('--out_dir', type=str, default="misc/10level2col/attention_visualizations_dinov2_252_cropped", help="Output directory")
    parser.add_argument('--num_images', type=int, default=20, help="Number of images to process")
    args = parser.parse_args()

    init_seed(42)

    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
        
    test_config_path = "./training/config/test_config.yaml"
    if os.path.exists(test_config_path):
        with open(test_config_path, "r") as f:
            config.update(yaml.safe_load(f))
            
    config["test_dataset"] = args.test_dataset

    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    
    ckpt = torch.load(args.weights_path, map_location=device)
    ckpt = {k.replace("module.backbone.", "backbone."): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    cfg_detector = config.get("degradation", {})
    degradation_config = {**cfg_detector, "use_beta": False, "distractor_p": 0.0, "op_p": 1.0, 
                          "mean": config.get("mean", [0.485, 0.456, 0.406]), "std": config.get("std", [0.229, 0.224, 0.225])}
    
    wrapper = NoiseAugmentWrapper(dataset, degradation_config, split="test")

    os.makedirs(args.out_dir, exist_ok=True)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    all_image_paths = sorted(dataset.data_dict['image'])

    print(f"Starting batch visualization for the first {args.num_images} images...")

    for idx in range(args.num_images):
        # --- NEW: Assign the index as the unique, deterministic seed for THIS image ---
        # This seed is passed directly into the NoiseAugmentWrapper
        wrapper.manual_seed = idx + 4200 

        image_path = all_image_paths[idx] 
        image_filename = os.path.basename(image_path)
        print(f"[{idx+1}/{args.num_images}] Processing: {image_filename} (Seed: {wrapper.manual_seed})")
        
        img_pil = dataset.load_rgb(image_path)
        img_np_base = np.array(img_pil).astype(np.float32) / 255.0

        fig, axes = plt.subplots(4, 6, figsize=(20, 13))
        fig.suptitle(f"Attention Collapse on Artifacts - {image_filename}", fontsize=18, fontweight='bold', y=0.98)

        for ax in axes.flat:
            ax.axis('off')

        for n_idx, noise in enumerate(noise_levels):
            if n_idx < 6:
                row_img, row_map, col = 0, 1, n_idx
            else:
                row_img, row_map, col = 2, 3, n_idx - 6

            if noise == 0.0:
                img_noisy = img_np_base.copy()
            else:
                wrapper.strength = noise
                img_noisy = wrapper.degradation_fn(img_np_base.copy())

            img_tensor = dataset.to_tensor(img_noisy)
            img_norm = dataset.normalize(img_tensor).unsqueeze(0).to(device)

            core_hf_model = model.get_backbone().vision_model
            if hasattr(core_hf_model, "base_model"): core_hf_model = core_hf_model.base_model
            if hasattr(core_hf_model, "model"): core_hf_model = core_hf_model.model
            if hasattr(core_hf_model.config, "_attn_implementation"): core_hf_model.config._attn_implementation = "eager"
            core_hf_model.config.output_attentions = True

            # --- NEW: Multi-Layer Attention Average ---
            with torch.no_grad():
                out = core_hf_model(pixel_values=img_norm, output_attentions=True, return_dict=True)
                
            # Stack the last 4 layers
            stacked_attentions = torch.stack(out.attentions[-4:]) # [4_layers, batch, num_heads, seq, seq]
            
            # Extract CLS token attention to all image patches (skip the CLS token itself)
            cls_attention = stacked_attentions[:, 0, :, 0, 1:] 
            
            # Average across both the 4 layers (dim=0) and all attention heads (dim=1)
            mean_attention = cls_attention.mean(dim=(0, 1)).cpu().numpy()

            grid_size = int(np.sqrt(mean_attention.shape[0]))
            heatmap = mean_attention.reshape(grid_size, grid_size)
            
            heatmap = heatmap - np.min(heatmap)
            heatmap = heatmap / np.max(heatmap)
            heatmap_resized = cv2.resize(heatmap, (img_noisy.shape[1], img_noisy.shape[0]))

            axes[row_img, col].imshow(img_noisy)
            axes[row_img, col].set_title(f"Noise: {noise:.1f}", fontsize=12)

            axes[row_map, col].imshow(img_noisy)
            axes[row_map, col].imshow(heatmap_resized, cmap='jet', alpha=0.5) 
            axes[row_map, col].set_title(f"Attention Map", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        save_path = os.path.join(args.out_dir, f"dinov2_attn_{idx:02d}_{image_filename.split('.')[0]}.png")
        plt.savefig(save_path, dpi=200) 
        plt.close(fig) 

    print(f"\n✅ All {args.num_images} images processed! Check the '{args.out_dir}/' folder.")

if __name__ == "__main__":
    main()