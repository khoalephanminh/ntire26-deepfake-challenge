import os
import random
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.noise_augment_wrapper import NoiseAugmentWrapper
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW: Deterministic Seed Function ---
def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    # 1. Lock everything down deterministically
    init_seed(42)

    # 1. Load Config (Using your pure DINOv2 global model)
    detector_path = "training/config/detector/dinov2_252.yaml"
    weights_path = "pretrained_weights/dinov2_252_crop.pth"
    
    with open(detector_path, "r") as f:
        config = yaml.safe_load(f)
        
    test_config_path = "./training/config/test_config.yaml"
    if os.path.exists(test_config_path):
        with open(test_config_path, "r") as f:
            config.update(yaml.safe_load(f))
            
    config["test_dataset"] = "publictest_data_final_cropped_original"

    # 2. Load Dataset & Model
    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    
    ckpt = torch.load(weights_path, map_location=device)
    ckpt = {k.replace("module.backbone.", "backbone."): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 3. Setup Noise Wrapper
    cfg_detector = config.get("degradation", {})
    degradation_config = {**cfg_detector, "use_beta": False, "distractor_p": 0.0, "op_p": 1.0, 
                          "mean": config.get("mean", [0.485, 0.456, 0.406]), "std": config.get("std", [0.229, 0.224, 0.225])}
    wrapper = NoiseAugmentWrapper(dataset, degradation_config, split="test")

    # --- NEW: Create output directory ---
    out_dir = "misc/10level2col/attention_visualizations_dinov2_252_cropped"
    os.makedirs(out_dir, exist_ok=True)
    
    num_images_to_process = 20
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Adjust as needed for more/less noise levels

    # --- NEW: Alphabetical Path Lock ---
    # Extract all paths, sort them alphabetically to guarantee OS-agnostic determinism
    all_image_paths = sorted(dataset.data_dict['image'])

    print(f"Starting batch visualization for the first {num_images_to_process} images...")

    # --- NEW: Loop through the first 20 images ---
    for idx in range(num_images_to_process):
        image_path = all_image_paths[idx] # Pull from the sorted list
        image_filename = os.path.basename(image_path)
        print(f"[{idx+1}/{num_images_to_process}] Processing: {image_filename}")
        
        img_pil = dataset.load_rgb(image_path)
        img_np_base = np.array(img_pil).astype(np.float32) / 255.0

        # Create a 4x6 grid. (6 columns is enough to fit half the sequence nicely)
        fig, axes = plt.subplots(4, 6, figsize=(20, 13))
        fig.suptitle(f"Attention Collapse on Artifacts - {image_filename}", fontsize=18, fontweight='bold', y=0.98)

        # Turn off all axes by default so the empty 12th slot (bottom right) stays blank and clean
        for ax in axes.flat:
            ax.axis('off')

        for idx, noise in enumerate(noise_levels):
            # Calculate where this step belongs in the wrapped grid
            if idx < 6:
                row_img, row_map, col = 0, 1, idx
            else:
                row_img, row_map, col = 2, 3, idx - 6

            # Apply Noise
            if noise == 0.0:
                img_noisy = img_np_base.copy()
            else:
                wrapper.strength = noise
                img_noisy = wrapper.degradation_fn(img_np_base.copy())

            # Forward Pass
            img_tensor = dataset.to_tensor(img_noisy)
            img_norm = dataset.normalize(img_tensor).unsqueeze(0).to(device)

            core_hf_model = model.get_backbone().vision_model
            if hasattr(core_hf_model, "base_model"): core_hf_model = core_hf_model.base_model
            if hasattr(core_hf_model, "model"): core_hf_model = core_hf_model.model
            if hasattr(core_hf_model.config, "_attn_implementation"): core_hf_model.config._attn_implementation = "eager"
            core_hf_model.config.output_attentions = True

            with torch.no_grad():
                out = core_hf_model(pixel_values=img_norm, output_attentions=True, return_dict=True)
                
            # Extract Attention
            attentions = out.attentions[-1]
            cls_attention = attentions[0, :, 0, 1:] 
            mean_attention = cls_attention.mean(dim=0).cpu().numpy()

            # Reshape to 2D grid
            grid_size = int(np.sqrt(mean_attention.shape[0]))
            heatmap = mean_attention.reshape(grid_size, grid_size)
            
            # Normalize and Resize heatmap
            heatmap = heatmap - np.min(heatmap)
            heatmap = heatmap / np.max(heatmap)
            heatmap_resized = cv2.resize(heatmap, (img_noisy.shape[1], img_noisy.shape[0]))

            # Top Row of the block: The Image
            axes[row_img, col].imshow(img_noisy)
            axes[row_img, col].set_title(f"Noise: {noise:.1f}", fontsize=12)

            # Bottom Row of the block: The Attention Overlay
            axes[row_map, col].imshow(img_noisy)
            axes[row_map, col].imshow(heatmap_resized, cmap='jet', alpha=0.5) 
            axes[row_map, col].set_title(f"Attention Map", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Prevent title overlap
        
        # --- NEW: Save dynamically and close figure ---
        save_path = os.path.join(out_dir, f"dinov2_attn_{idx:02d}_{image_filename.split('.')[0]}.png")
        plt.savefig(save_path, dpi=200) # Slightly lower DPI to save time/space for 20 images
        plt.close(fig) # CRITICAL: Prevents Memory Leak!

    print(f"\n✅ All 20 images processed! Check the '{out_dir}/' folder.")

if __name__ == "__main__":
    main()