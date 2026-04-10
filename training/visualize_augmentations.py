import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import string
import random
import matplotlib.font_manager as fm

def seed_everything(seed=42):
    """Locks all random seeds for complete reproducibility."""
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy random state
    np.random.seed(seed)
    
    # 4. OpenCV random number generator
    cv2.setRNGSeed(seed)
    
    # Optional: If you ever add PyTorch to this file later, uncomment these:
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# ==========================================
# 1. CVPR FORMATTING SETUP (Native Windows Fonts)
# ==========================================
font_paths = [
    "/raid/dtle/ntire26-deepfake-challenge/fonts/TIMES.TTF",
    "/raid/dtle/ntire26-deepfake-challenge/fonts/TIMESBD.TTF",
    "/raid/dtle/ntire26-deepfake-challenge/fonts/TIMESBI.TTF",
    "/raid/dtle/ntire26-deepfake-challenge/fonts/TIMESI.TTF"
]

for path in font_paths:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
    else:
        print(f"Warning: Font missing at {path}")

# Force the exact font family name found in TIMES.TTF
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['pdf.fonttype'] = 42          
plt.rcParams['ps.fonttype'] = 42           
plt.rcParams['axes.linewidth'] = 1.0       
# ==========================================

# Import all your custom degradation functions
from dataset.noise_augment_wrapper_001 import (
    add_smoothing2, add_resize, add_Gaussian_noise, add_speckle_noise,
    add_Poisson_noise, add_JPEG_noise, add_salt_and_pepper,
    add_color_cast_and_fade, add_grayscale_drop, add_color_banding,
    add_chroma_subsampling, add_motion_blur, add_enhance_advanced,
    add_chromatic_aberration, add_moire_pattern, add_glitch_and_shift,
    add_vignetting
)

# Import the random mix function
from dataset.customize_noise_augment_001 import apply_targeted_degradation

def add_text_distractor(img_np):
    """Simulates the text distractor logic from operation 14 in the wrapper."""

    res = img_np.copy()
    h, w = res.shape[:2]
    color = (random.random(), random.random(), random.random())
    text_len = random.randint(6, 12)
    # Get a random string, avoiding weird whitespace characters
    text = "".join(np.random.choice(list(string.ascii_letters + string.digits), text_len, replace=True))
    
    # Scale to uint8 for cv2 text drawing, then back to float32
    res_uint8 = (res * 255).astype(np.uint8)
    res_uint8 = cv2.putText(
        img = res_uint8,
        text = text, 
        org = (random.randint(10, w // 3), random.randint(h // 3, h - 20)), 
        fontFace = np.random.randint(8), 
        fontScale = random.uniform(2.0, 3), 
        # color = tuple(int(c * 255) for c in color), 
        #set red
        color = (255, 0, 0),
        thickness = random.randint(2, 5),
        lineType = cv2.LINE_AA
    )
    return (res_uint8.astype(np.float32) / 255.0)

def load_image_as_float(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Cannot find image: {img_path}")
    img_pil = Image.open(img_path).convert("RGB")
    # Resize to 256x256 for a perfectly uniform square grid
    img_pil = img_pil.resize((256, 256), Image.Resampling.BICUBIC)
    return np.array(img_pil).astype(np.float32) / 255.0

def generate_augmentations(img_np):
    """
    Applies high-severity versions of each operation.
    CRITICAL 1-COLUMN FIX: Dictionary keys use \n to stack text vertically,
    preventing massive titles from bleeding into neighboring images.
    """
    aug_dict = {"Original\nImage": img_np.copy()}
    
    # 1. Optical / Blur Artifacts
    aug_dict["Gaussian\nBlur"] = add_smoothing2(img_np.copy(), strength=1.0)
    aug_dict["Motion\nBlur"] = add_motion_blur(img_np.copy(), strength=1.0)
    aug_dict["Chrom.\nAberration"] = add_chromatic_aberration(img_np.copy(), strength=1.0)
    aug_dict["Lens\nVignetting"] = add_vignetting(img_np.copy(), strength=1.0)
    
    # 2. Sensor / Digital Noise
    aug_dict["Gaussian\nNoise"] = add_Gaussian_noise(img_np.copy(), noise_level1=50, noise_level2=50, strength=1.0)
    aug_dict["Speckle\nNoise"] = add_speckle_noise(img_np.copy(), noise_level1=50, noise_level2=50, strength=1.0)
    aug_dict["Poisson\nNoise"] = add_Poisson_noise(img_np.copy())
    aug_dict["Salt &\nPepper"] = add_salt_and_pepper(img_np.copy(), prob=0.05, strength=1.0)
    
    # 3. Compression / Resampling
    aug_dict["JPEG\n(Q=10)"] = add_JPEG_noise(img_np.copy(), bounds=(10, 10))
    aug_dict["Chroma\nSubsample"] = add_chroma_subsampling(img_np.copy())
    aug_dict["Color\nBanding"] = add_color_banding(img_np.copy(), strength=1.0)
    aug_dict["Aggressive\nResize"] = add_resize(img_np.copy(), sf=2, strength=1.0)
    
    # 4. Photometric / Transmission Distortions
    aug_dict["Grayscale\nDrop"] = add_grayscale_drop(img_np.copy())
    aug_dict["Color\nCast/Fade"] = add_color_cast_and_fade(img_np.copy(), strength=1.0)
    aug_dict["Bright /\nContrast"] = add_enhance_advanced(img_np.copy(), strength=1.0)
    aug_dict["Moiré\nPattern"] = add_moire_pattern(img_np.copy(), strength=1.0)
    
    # 5. Visual Distractor & Compound
    aug_dict["Glitch &\nShift"] = add_glitch_and_shift(img_np.copy(), strength=1.0)
    aug_dict["Text\nDistractor"] = add_text_distractor(img_np.copy())
    aug_dict["Compound\nMix"] = apply_targeted_degradation(img_np.copy(), "random_mix", 0.8, seed=50)

    return aug_dict

def plot_augmentations(aug_dict, output_path="augmentation_grid.pdf"):
    # EXACTLY 20 images = Perfect 4 Columns x 5 Rows Grid
    cols = 5
    rows = 4
    
    # Taller figsize (8x11.5) to accommodate the stacked 2-line titles
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8.5))
    axes = axes.flatten()
    
    for idx, (title, img) in enumerate(aug_dict.items()):
        ax = axes[idx]
        img_display = np.clip(img, 0.0, 1.0)
        ax.imshow(img_display)
        
        # JUMBO Font size for 1-column scaling. 
        # ha='center' keeps the stacked text perfectly aligned over the image.
        single_line_title = title.replace('\n', ' ')
        ax.set_title(single_line_title, fontsize=14, fontweight='bold', pad=2, ha='center')
        ax.axis("off")
        
    # Extremely tight spacing. wspace=0.02 pushes images together horizontally.
    # hspace=0.35 gives just enough vertical room for the 2-line titles.
    plt.subplots_adjust(wspace=0.02, hspace=0.15)
    
    # Save as high-res PDF (best for LaTeX) and PNG with zero redundant white space
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved ultra-compact 1-column visualization grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Extreme Compound Degradations")
    parser.add_argument("--image_path", type=str, default="/raid/dtle/NTIRE26-DeepfakeDetection/datasets/rgb/HIDF/Fake-img/c00014_c00004.jpg", help="Path to a clean image for augmentation")
    parser.add_argument("--output", type=str, default="figures_001/augmentation_grid.pdf", help="Path to save the output grid")
    args = parser.parse_args()

    # ==========================================
    # LOCK THE SEED HERE BEFORE ANY IMAGES LOAD
    # ==========================================
    seed_everything(seed=42)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        clean_img = load_image_as_float(args.image_path)
        augmented_samples = generate_augmentations(clean_img)
        plot_augmentations(augmented_samples, args.output)
    except Exception as e:
        print(f"Error: {e}")