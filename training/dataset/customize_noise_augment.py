# customize_noise_augment.py
import numpy as np
import random
import cv2

# Import your base operations from your existing file
# Adjust this import to match your actual file name!
from training.dataset.noise_augment_wrapper import (
    add_JPEG_noise, add_Gaussian_noise, add_smoothing2, 
    add_motion_blur, add_speckle_noise, add_Poisson_noise,
    add_chroma_subsampling, add_color_banding, add_chromatic_aberration,
    add_moire_pattern, add_glitch_and_shift, add_vignetting
)

def apply_targeted_degradation(img_np, op_name, op_level, seed=None):
    """
    Applies a specific degradation at a specific level.
    img_np: numpy array (H, W, 3) in [0.0, 1.0]
    op_name: str, name of degradation
    op_level: float or int, severity
    seed: int, used to make 'random_mix' deterministic across different models
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    img = img_np.copy()

    # --- ADD THIS SHORT-CIRCUIT FIX ---
    # If the operation is not JPEG, and the level is 0, just return the clean image!
    # (JPEG uses 100 for clean, 0 for max degradation, so we skip this check for jpeg)
    if op_name != "jpeg" and op_level <= 0.0:
        return img
    # ----------------------------------
    
    if op_name == "clean":
        return img
        
    elif op_name == "jpeg":
        # op_level is quality factor (e.g., 100 to 0)
        lvl = max(1, int(op_level))
        img = add_JPEG_noise(img, bounds=(lvl, lvl))
        
    elif op_name == "gaussian_noise":
        # op_level is noise standard deviation (e.g., 0 to 60)
        lvl = max(1, int(op_level))
        img = add_Gaussian_noise(img, noise_level1=lvl, noise_level2=lvl, strength=1.0)
        
    elif op_name == "gaussian_blur":
        # op_level is strength factor (e.g., 0.0 to 1.0)
        img = add_smoothing2(img, strength=op_level)
        
    elif op_name == "motion_blur":
        # op_level is strength factor (e.g., 0.0 to 1.0)
        img = add_motion_blur(img, strength=op_level)
        
    elif op_name == "random_mix":
        # Replicate a deterministic version of your 15-op compound degradation
        # op_level dictates the severity of ALL applied operations
        n_cycles = random.randint(1, 3)
        for _ in range(n_cycles):
            if random.random() < 0.5:
                img = add_JPEG_noise(img, bounds=(int(30 + 70*(1-op_level)), 100))
        
        shuffle_order = random.sample(range(10), 8) # Pick 5 random ops
        for i in shuffle_order:
            if i == 0: img = add_smoothing2(img, strength=op_level)
            elif i == 1: img = add_motion_blur(img, strength=op_level)
            elif i == 2: img = add_Gaussian_noise(img, noise_level1=2, noise_level2=int(60*op_level)+2, strength=1)
            elif i == 3: img = add_chroma_subsampling(img)
            elif i == 4: img = add_chromatic_aberration(img, strength=op_level)
            elif i == 5: img = add_moire_pattern(img, strength=op_level)
            elif i == 6: img = add_glitch_and_shift(img, strength=op_level)
            elif i == 7: img = add_vignetting(img, strength=op_level)
            elif i == 8: img = add_speckle_noise(img, strength=op_level)
            elif i == 9: img = add_color_banding(img, strength=op_level)
            
    return np.clip(img, 0.0, 1.0)