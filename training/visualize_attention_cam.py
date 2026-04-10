import os
import random
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import torch
import torch.backends.cudnn as cudnn

# Import CAM libraries
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

# --- CAM UTILITIES ---
class FSBIDetectorWrapper(torch.nn.Module):
    """Wraps FSBIDetector so CAM can pass raw tensors in and get logits out."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mock_dict = {
            'image': x, 
            'label': torch.zeros(x.shape[0], dtype=torch.long).to(x.device) 
        }
        pred_dict = self.model(mock_dict, inference=False) 
        return pred_dict['cls']

def dinov2_reshape_transform(tensor):
    """Strips the [CLS] token and reshapes the 1D sequence into a 2D grid."""
    sequence_length = tensor.shape[1]
    num_patches = sequence_length - 1 
    grid_size = int(np.sqrt(num_patches))
    result = tensor[:, 1:, :].reshape(tensor.shape[0], grid_size, grid_size, tensor.shape[2])
    result = result.permute(0, 3, 1, 2)
    return result

def main():
    parser = argparse.ArgumentParser(description="Visualize Class Attribution (CAM) under Degradation")
    parser.add_argument('--detector_path', type=str, default="training/config/detector/dinov2_252.yaml")
    parser.add_argument('--weights_path', type=str, default="pretrained_weights/dinov2_252_crop.pth")
    parser.add_argument('--test_dataset', type=str, default="publictest_data_final_cropped_original")
    parser.add_argument('--out_dir', type=str, default="misc/scorecam_visualizations")
    parser.add_argument('--num_images', type=int, default=10) #13
    parser.add_argument('--cam_method', type=str, default="ScoreCAM", help="ScoreCAM, EigenCAM, GradCAM, etc.")
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

    # CAM Setup
    cam_model = FSBIDetectorWrapper(model).eval()
    target_layers = [model.backbone.vision_model.encoder.layer[-1].norm1]
    cam_targets = [ClassifierOutputTarget(1)] # Target class 1 (Fake)

    cfg_detector = config.get("degradation", {})
    degradation_config = {**cfg_detector, "use_beta": False, "distractor_p": 0.0, "op_p": 1.0, 
                          "mean": config.get("mean", [0.485, 0.456, 0.406]), "std": config.get("std", [0.229, 0.224, 0.225])}
    
    wrapper = NoiseAugmentWrapper(dataset, degradation_config, split="test")

    os.makedirs(args.out_dir, exist_ok=True)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    all_image_paths = sorted(dataset.data_dict['image'])

    print(f"Starting batch visualization for the first {args.num_images} images using {args.cam_method}...")

    # for idx in range(968, 968 + args.num_images):
    for idx in range(10, 10 + args.num_images):
        wrapper.manual_seed = idx + 4200 

        image_path = all_image_paths[idx] 
        image_filename = os.path.basename(image_path)
        img_base_name = image_filename.split('.')[0]
        print(f"[{idx+1}/{args.num_images}] Processing: {image_filename}")

        # --- FOLDER SETUP FOR INDIVIDUAL ASSETS ---
        ind_dir = os.path.join(args.out_dir, "individuals", img_base_name)
        os.makedirs(os.path.join(ind_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "scorecam"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "scorecam_overlay"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "raw_cam_npy"), exist_ok=True) # For perfect entropy math later
        os.makedirs(os.path.join(ind_dir, "probabilities"), exist_ok=True)
        
        img_pil = dataset.load_rgb(image_path)
        img_np_base = np.array(img_pil).astype(np.float32) / 255.0

        # --- 4x6 GRID FOR COMBINED PLOT (No CLS anymore) ---
        fig, axes = plt.subplots(4, 6, figsize=(20, 13))
        fig.suptitle(f"Attribution Collapse ({args.cam_method}) - {image_filename}", fontsize=18, fontweight='bold', y=0.98)

        for ax in axes.flat:
            ax.axis('off')

        for n_idx, noise in enumerate(noise_levels):
            # 2 Rows per block now
            if n_idx < 6:
                row_img, row_cam, col = 0, 1, n_idx
            else:
                row_img, row_cam, col = 2, 3, n_idx - 6

            if noise == 0.0:
                img_noisy = img_np_base.copy()
            else:
                wrapper.strength = noise
                img_noisy = wrapper.degradation_fn(img_np_base.copy())
            
            img_noisy_clipped = np.clip(img_noisy, 0.0, 1.0)
            img_tensor = dataset.to_tensor(img_noisy)
            img_norm = dataset.normalize(img_tensor).unsqueeze(0).to(device)

            # --- NEW: Fast Inference for Prediction Probability ---
            with torch.no_grad():
                mock_dict = {
                    'image': img_norm, 
                    'label': torch.zeros(1, dtype=torch.long).to(device)
                }
                pred_dict = model(mock_dict, inference=True)
                # Extract the float probability of being Fake
                prob_fake = float(pred_dict['prob'][0].cpu().item())
            
            noise_str = f"{noise:.1f}"
            
            # --- NEW: Save Probability to TXT ---
            prob_txt_path = os.path.join(ind_dir, "probabilities", f"{img_base_name}_{noise_str}.txt")
            with open(prob_txt_path, "w") as f:
                f.write(f"{prob_fake:.6f}\n")
            # ------------------------------------
            
            # --- CAM EXTRACTION ---
            cam_class = getattr(pytorch_grad_cam, args.cam_method)
            with cam_class(model=cam_model, target_layers=target_layers, reshape_transform=dinov2_reshape_transform) as cam_instance:
                if args.cam_method in ["ScoreCAM", "AblationCAM"]:
                    cam_instance.batch_size = 128 # Adjust to fit your VRAM (64 if capable)
                
                # grayscale_cam is a 2D float array in [0, 1]
                grayscale_cam = cam_instance(input_tensor=img_norm, targets=cam_targets)[0, :]
                
                # overlay is a 3D uint8 RGB array
                cam_overlay = show_cam_on_image(img_noisy_clipped, grayscale_cam, use_rgb=True)

            # --- SAVE INDIVIDUAL ASSETS ---
            noise_str = f"{noise:.1f}"
            
            # 1. Original (Convert RGB Float -> BGR uint8)
            orig_bgr = cv2.cvtColor((img_noisy_clipped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ind_dir, "original", f"{img_base_name}_{noise_str}.jpg"), orig_bgr)
            
            # 2. Pure ScoreCAM (Convert Float -> Jet Colormap BGR)
            heatmap_uint8 = np.uint8(255 * grayscale_cam)
            pure_cam_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(ind_dir, "scorecam", f"{img_base_name}_{noise_str}.jpg"), pure_cam_bgr)
            
            # 3. Overlay (Convert RGB -> BGR)
            overlay_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ind_dir, "scorecam_overlay", f"{img_base_name}_{noise_str}.jpg"), overlay_bgr)

            # 4. Raw Math Array (For Entropy Calculation)
            np.save(os.path.join(ind_dir, "raw_cam_npy", f"{img_base_name}_{noise_str}.npy"), grayscale_cam)

            # --- PLOT COMBINED GRID ---
            axes[row_img, col].imshow(img_noisy_clipped)
            axes[row_img, col].set_title(f"Noise: {noise:.1f}", fontsize=12)

            axes[row_cam, col].imshow(cam_overlay)
            axes[row_cam, col].set_title(f"{args.cam_method}", fontsize=10)

            torch.cuda.empty_cache() 

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        # Save the combined mega-grid
        save_path = os.path.join(args.out_dir, f"combined_{args.cam_method}_{idx:02d}_{img_base_name}.png")
        plt.savefig(save_path, dpi=200) 
        plt.close(fig) 

    print(f"\n✅ All {args.num_images} images processed! Check the '{args.out_dir}/' folder.")

if __name__ == "__main__":
    main()