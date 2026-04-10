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
    parser = argparse.ArgumentParser(description="Targeted CAM Extraction (Noise 0.0)")
    parser.add_argument('--detector_path', type=str, default="training/config/detector/dinov2_252.yaml")
    parser.add_argument('--weights_path', type=str, default="pretrained_weights/dinov2_252_crop.pth")
    parser.add_argument('--test_dataset', type=str, default="publictest_data_final_cropped_original")
    parser.add_argument('--out_dir', type=str, default="misc/targeted_rescue_cams")
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

    os.makedirs(args.out_dir, exist_ok=True)
    all_image_paths = sorted(dataset.data_dict['image'])

    # --- THE TARGETED LIST (1-based converted to 0-based) ---
    target_indices_1_based = [26, 53, 99, 117, 126, 148, 158, 171, 186, 252, 295, 386, 477, 478, 503, 523, 543, 566, 645, 661, 686, 691, 750, 789, 821, 823, 874, 959, 964]
    target_indices = [idx for idx in target_indices_1_based]

    print(f"Starting targeted visualization for {len(target_indices)} specific images at Noise 0.0 using {args.cam_method}...")

    for i, idx in enumerate(target_indices):
        image_path = all_image_paths[idx] 
        image_filename = os.path.basename(image_path)
        img_base_name = image_filename.split('.')[0]
        print(f"[{i+1}/{len(target_indices)}] Processing Index {idx} (Original {idx+1}): {image_filename}")

        # --- FOLDER SETUP FOR INDIVIDUAL ASSETS ---
        ind_dir = os.path.join(args.out_dir, "individuals", img_base_name)
        os.makedirs(os.path.join(ind_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "scorecam"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "scorecam_overlay"), exist_ok=True)
        os.makedirs(os.path.join(ind_dir, "raw_cam_npy"), exist_ok=True) 
        os.makedirs(os.path.join(ind_dir, "probabilities"), exist_ok=True)
        
        # Load and prep the clean image (Noise 0.0)
        img_pil = dataset.load_rgb(image_path)
        img_np_base = np.array(img_pil).astype(np.float32) / 255.0
        img_noisy_clipped = np.clip(img_np_base, 0.0, 1.0) # Just clean base
        
        img_tensor = dataset.to_tensor(img_noisy_clipped)
        img_norm = dataset.normalize(img_tensor).unsqueeze(0).to(device)

        # --- Fast Inference for Prediction Probability ---
        with torch.no_grad():
            mock_dict = {
                'image': img_norm, 
                'label': torch.zeros(1, dtype=torch.long).to(device)
            }
            pred_dict = model(mock_dict, inference=True)
            prob_fake = float(pred_dict['prob'][0].cpu().item())
        
        noise_str = "0.0"
        
        # --- Save Probability to TXT ---
        prob_txt_path = os.path.join(ind_dir, "probabilities", f"{img_base_name}_{noise_str}.txt")
        with open(prob_txt_path, "w") as f:
            f.write(f"{prob_fake:.6f}\n")

        # --- CAM EXTRACTION ---
        cam_class = getattr(pytorch_grad_cam, args.cam_method)
        with cam_class(model=cam_model, target_layers=target_layers, reshape_transform=dinov2_reshape_transform) as cam_instance:
            if args.cam_method in ["ScoreCAM", "AblationCAM"]:
                cam_instance.batch_size = 128
            
            grayscale_cam = cam_instance(input_tensor=img_norm, targets=cam_targets)[0, :]
            cam_overlay = show_cam_on_image(img_noisy_clipped, grayscale_cam, use_rgb=True)

        # --- SAVE INDIVIDUAL ASSETS ---
        orig_bgr = cv2.cvtColor((img_noisy_clipped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(ind_dir, "original", f"{img_base_name}_{noise_str}.jpg"), orig_bgr)
        
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        pure_cam_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(ind_dir, "scorecam", f"{img_base_name}_{noise_str}.jpg"), pure_cam_bgr)
        
        overlay_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(ind_dir, "scorecam_overlay", f"{img_base_name}_{noise_str}.jpg"), overlay_bgr)

        np.save(os.path.join(ind_dir, "raw_cam_npy", f"{img_base_name}_{noise_str}.npy"), grayscale_cam)

        # --- PLOT CLEAN 1x2 SIDE-BY-SIDE ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Image {idx+1} | {args.cam_method} | P(Fake): {prob_fake:.4f}", fontsize=14, fontweight='bold')

        axes[0].imshow(img_noisy_clipped)
        axes[0].set_title("Original (Clean)", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(cam_overlay)
        axes[1].set_title(f"Attribution Overlay", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f"rescue_{idx+1:04d}_{img_base_name}.png")
        plt.savefig(save_path, dpi=200) 
        plt.close(fig) 
        
        torch.cuda.empty_cache() 

    print(f"\n✅ All {len(target_indices)} targeted images processed! Check the '{args.out_dir}/' folder.")

if __name__ == "__main__":
    main()