# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import random
import scipy.stats as ss
from scipy import ndimage
from scipy.linalg import orth
from PIL import Image, ImageEnhance
import logging
import string  # Added missing import

# Make sure this import path matches your repository structure
from training.dataset.library.pdm import utils_image as util

# ====================================================================
# CUSTOM EXTREME DEGRADATION HELPERS (Original + New)
# ====================================================================

def add_salt_and_pepper(img, prob=0.05, strength=1):
    actual_prob = prob * strength
    # Changed *img.shape[:2] to *img.shape to apply independently to R, G, B channels
    rnd = np.random.rand(*img.shape)
    img[rnd < (actual_prob / 2)] = 0.0
    img[rnd > 1 - (actual_prob / 2)] = 1.0
    return img

def add_color_cast_and_fade(img, strength=1):
    r = random.random()
    if r < 0.3: 
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        converter = ImageEnhance.Color(img_pil)
        img = np.array(converter.enhance(random.uniform(0.1, max(0.2, 1.0 - strength)))) / 255.0
    else: 
        cast = np.random.uniform(1 - (0.4 * strength), 1 + (0.4 * strength), 3)
        if random.random() < 0.2: 
            cast = np.array([0.8, 1.2, 0.8])
        img = img * cast
    return np.clip(img, 0.0, 1.0)

def add_grayscale_drop(img):
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    return img

def add_color_banding(img, strength=1):
    if random.random() < strength: 
        return img
    levels = int(max(4, 32 - (28 * strength * 0.5))) 
    factor = 255.0 / (levels - 1)
    img = np.round((img * 255.0) / factor) * factor / 255.0
    return np.clip(img, 0.0, 1.0)

def add_chroma_subsampling(img):
    if img.shape[0] < 2 or img.shape[1] < 2:
        return img
    img_uint = (img * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_uint, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    w_sub, h_sub = max(1, cr.shape[1] // 2), max(1, cr.shape[0] // 2)
    cr_sub = cv2.resize(cr, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
    cb_sub = cv2.resize(cb, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
    cr_up = cv2.resize(cr_sub, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_NEAREST)
    cb_up = cv2.resize(cb_sub, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_NEAREST)
    res = cv2.cvtColor(cv2.merge([y, cr_up, cb_up]), cv2.COLOR_YCrCb2RGB)
    return (res.astype(np.float32) / 255.0)
    
def add_motion_blur(img, strength=1):
    size = int(max(3, 15 * strength * 0.8)) 
    if size % 2 == 0: size += 1
    kernel = np.zeros((size, size))
    angle = random.uniform(0, 180)
    center = size // 2
    slope = np.tan(np.radians(angle))
    for x in range(size):
        y = int(slope * (x - center) + center)
        if 0 <= y < size:
            kernel[y, x] = 1
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return np.clip(img, 0.0, 1.0)

def add_enhance_advanced(img, strength=1):
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    r = random.random()
    if r < 0.45:
        enhance = ImageEnhance.Brightness(img)
        img = enhance.enhance(random.uniform(max(0.02, 1.0 - (0.98 * strength)), 1.0))
    elif r < 0.90:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0, 1.0 + (1.5 * strength)))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(max(0.1, 1.0 - (0.9 * strength)), 0.8))
    else:
        enhance = random.choice([ImageEnhance.Brightness(img), ImageEnhance.Contrast(img)])
        img = enhance.enhance(random.uniform(max(0.5, 1.0 - 0.5 * strength), 1.0 + 0.5 * strength))
    return np.array(img).astype(float) / 255.0

# --- NEW: ADVANCED REAL-WORLD ARTIFACTS ---

def add_chromatic_aberration(img, strength=1):
    """Simulates cheap lens color fringing (RGB shift)."""
    shift = int(max(1, 4 * strength))
    res = np.copy(img)
    # Shift Red channel down-right, Blue channel up-left
    res[shift:, shift:, 0] = img[:-shift, :-shift, 0]
    res[:-shift, :-shift, 2] = img[shift:, shift:, 2]
    return np.clip(res, 0.0, 1.0)

def add_moire_pattern(img, strength=1):
    """Simulates taking a photo of a digital screen (Moiré interference)."""
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w, w), np.linspace(0, h, h))
    frequency = random.uniform(0.5, 3.0)
    angle = random.uniform(0, np.pi)
    # Inline formula: sine wave grid
    wave = np.sin((X * np.cos(angle) + Y * np.sin(angle)) * frequency)
    wave = (wave + 1) / 2.0 
    alpha = 0.15 * strength
    res = img * (1 - alpha) + wave[..., np.newaxis] * alpha
    return np.clip(res, 0.0, 1.0)

def add_glitch_and_shift(img, strength=1):
    """Simulates transmission packet loss and H.264 video glitches."""
    h, w = img.shape[:2]
    res = np.copy(img)
    num_lines = int(10 * strength)
    for _ in range(num_lines):
        y = random.randint(0, h-1)
        thickness = random.randint(1, max(2, int(8 * strength)))
        if y + thickness >= h: continue
        
        if random.random() < 0.4:
            # Drop block (black/white/gray band)
            res[y:y+thickness, :] = random.choice([0.0, 1.0, random.uniform(0.2, 0.8)])
        else:
            # Horizontal roll (pixel shifting)
            shift = random.randint(-int(30 * strength), int(30 * strength))
            res[y:y+thickness, :] = np.roll(res[y:y+thickness, :], shift, axis=1)
    return np.clip(res, 0.0, 1.0)

def add_vignetting(img, strength=1):
    """Simulates lens darkening at the corners to alter global gradients."""
    h, w = img.shape[:2]
    X = np.linspace(-1, 1, w)
    Y = np.linspace(-1, 1, h)
    U, V = np.meshgrid(X, Y)
    radius = np.sqrt(U**2 + V**2)
    # Create mask where center is 1.0 and edges fade down based on strength
    mask = np.clip(1.0 - (radius / np.sqrt(2)) * (0.4 + 0.4 * strength), 0, 1)
    return np.clip(img * mask[..., np.newaxis], 0.0, 1.0)

# ====================================================================
# BSRGAN FILTERS AND DEGRADATION MODELS 
# ====================================================================

def add_smoothing2(img, strength=1):
    wd2, wd = 30 * strength, 15 * strength
    r = random.random()
    if r < 1/3:
        l1, l2 = wd2*random.random(), wd2*random.random()
        k = anisotropic_Gaussian(ksize=int(max(l1, l2))*2+1, theta=random.random()*np.pi, l1=l1, l2=l2)
    elif r < 2/3:
        k = fspecial_gaussian(2*random.randint(2,11)+3, wd2*random.random())
    else:
        upper_bound = max(3, int(30 * strength))
        s = random.randint(3, upper_bound)
        k = np.ones((s, s)) / s**2
    return ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

def add_resize(img, sf=4, strength=1):
    rnum = np.random.rand()
    if rnum > 0.8: sf1 = random.uniform(1, 2)
    elif rnum < 0.7: sf1 = np.clip(random.uniform(0.5/sf/strength, 1), 0.1, 1)
    else: sf1 = 1.0

    interp = cv2.INTER_NEAREST if (sf1 < 1.0 and random.random() < 0.5) else random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    img = cv2.resize(img, (max(1, int(sf1*img.shape[1])), max(1, int(sf1*img.shape[0]))), interpolation=interp)
    return np.clip(img, 0.0, 1.0)

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            k[y, x] = ss.multivariate_normal.pdf([x - center + 1, y - center + 1], mean=mean, cov=cov)
    return k / np.sum(k)

def fspecial_gaussian(hsize, sigma):
    siz = [(hsize-1.0)/2.0, (hsize-1.0)/2.0]
    x, y = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    h = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    h[h < 1e-8 * h.max()] = 0
    sumh = h.sum()
    return h/sumh if sumh != 0 else h

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25, strength=1):
    noise_level = random.randint(noise_level1, noise_level2) * strength
    rnum = np.random.rand()
    if rnum < 0.70: 
        # Standard independent RGB noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    else:
        # Correlated RGB noise (still colored, but mimics realistic hardware sensor noise)
        L = noise_level2/255. * strength
        conv = np.dot(np.dot(np.transpose(orth(np.random.rand(3,3))), np.diag(np.random.rand(3))), orth(np.random.rand(3,3)))
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    return np.clip(img, 0.0, 1.0)

def add_speckle_noise(img, noise_level1=2, noise_level2=25, strength=1):
    noise_level = random.randint(noise_level1, noise_level2) * strength
    rnum = random.random()
    if rnum > 0.4: 
        # Independent RGB speckle
        img += img * np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    else:
        # Correlated RGB speckle
        L = noise_level2/255. * strength
        conv = np.dot(np.dot(np.transpose(orth(np.random.rand(3,3))), np.diag(np.random.rand(3))), orth(np.random.rand(3,3)))
        img += img * np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    return np.clip(img, 0.0, 1.0)

def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)
    # Always apply independent RGB poisson noise
    img = np.random.poisson(img * vals).astype(np.float32) / vals
    return np.clip(img, 0.0, 1.0)

def add_JPEG_noise(img, bounds=(10, 95)):
    quality_factor = random.randint(*bounds)
    img_bgr = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.cvtColor(util.uint2single(cv2.imdecode(encimg, 1)), cv2.COLOR_BGR2RGB)
    return img

# ====================================================================
# NOISE AUGMENTATION WRAPPER
# ====================================================================

class NoiseAugmentWrapper:
    def __init__(self, dataset, config, split="train"):
        print("Using NoiseAugmentWrapper, with config:", config)
        self.dataset = dataset
        self.global_p = config.get("global_p", 0.8) if split == "train" else config.get("val_global_p", 0.0)
        self.op_p = config.get("op_p", 0.7)
        self.strength = config["degradations_strength"]
        self.distractor_p = config.get("distractor_p", 0.5)
        self.use_beta = config.get("use_beta", False)
        if self.use_beta:
            self.a, self.b = config["a"], config["b"]

        self.degradation_fn = lambda x: self.degradation(x, op_p=self.op_p)

        self._norm = {}
        def _t(v): return torch.tensor(v, dtype=torch.float32).view(3, 1, 1)

        self._norm[0] = (_t(config["mean"]), _t(config["std"]))
        self._norm[1] = (_t(config.get("mean_2", config["mean"])), _t(config.get("std_2", config["std"])))
        self._norm[2] = (_t(config.get("mean_3", config["mean"])), _t(config.get("std_3", config["std"])))

        # --- NEW: Allow an external script to lock the seed per-image ---
        self.manual_seed = None

    def get_strength(self):
        return np.random.beta(self.a, self.b) if self.use_beta else self.strength

    def __getitem__(self, i):
        x = self.dataset[i]
        return self.degrade(x) if np.random.rand() < self.global_p else x

    def degrade(self, x):
        def _is_rgb_chw(t): return isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[0] == 3
        def _is_rgb_nchw(t): return isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[1] == 3

        if isinstance(x, torch.Tensor):
            return self.transpose_degrade_transpose(x, *self._norm[0]) if (_is_rgb_chw(x) or _is_rgb_nchw(x)) else x
        if isinstance(x, np.ndarray):
            return self.transpose_degrade_transpose(x, *self._norm[0]) if (x.ndim == 3 and x.shape[0] == 3) else x
        if isinstance(x, list):
            if len(x) in (2, 3) and all(isinstance(t, torch.Tensor) for t in x):
                return [self.transpose_degrade_transpose(t, *self._norm[i]) if _is_rgb_chw(t) or _is_rgb_nchw(t) else t for i, t in enumerate(x)]
            return [self.degrade(z) for z in x]
        if isinstance(x, tuple):
            return (self.degrade(x[0]), x[1], x[2], x[3]) if len(x) == 4 else tuple(self.degrade(z) for z in x)
        if isinstance(x, dict):
            return {k: self.degrade(v) for k, v in x.items()}
        return x

    def transpose_degrade_transpose(self, x, mean, std):
        if isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3:
            return torch.stack([self.transpose_degrade_transpose(i, mean, std) for i in x], dim=0)

        orig_dtype = x.dtype if isinstance(x, torch.Tensor) else None
        x_f = x.detach().cpu().to(torch.float32) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(torch.float32)
        if not (x_f.ndim == 3 and x_f.shape[0] == 3): return x 

        x01 = (x_f * std.cpu() + mean.cpu()).clamp(0.0, 1.0)
        img_hwc = x01.permute(1, 2, 0).numpy().astype(np.float32)
        out_hwc = self.degradation_fn(img_hwc)
        out_chw = torch.from_numpy(out_hwc).permute(2, 0, 1).contiguous().to(torch.float32)
        
        out_norm = (out_chw - mean.cpu()) / std.cpu()
        return out_norm.to(orig_dtype) if orig_dtype in (torch.float16, torch.bfloat16) else out_norm

    def __getattr__(self, i): return getattr(self.dataset, i)
    def __len__(self): return len(self.dataset)

    def degradation(self, img, op_p=0.5):
        # --- NEW: Image-Specific Deterministic Lock ---
        # If the visualizer passed a manual seed, lock it now.
        # This guarantees spatial consistency across noise scales!
        if self.manual_seed is not None:
            random.seed(self.manual_seed)
            np.random.seed(self.manual_seed)
            
        h, w, _ = img.shape

        if random.random() < op_p:
            n_cycles = random.randint(1, 5) 
            start_with_jpeg = (random.random() < 0.5)
            prep_strength = self.get_strength()

            for t in range(2 * n_cycles):
                do_jpeg = (t % 2 == 0) if start_with_jpeg else (t % 2 == 1)
                if do_jpeg:
                    img = add_JPEG_noise(img)
                else:
                    if img.shape[0] > 10 and img.shape[1] > 10:
                        img = add_resize(img, sf=2, strength=prep_strength)
                img = np.clip(img, 0.0, 1.0).astype(np.float32)
        
        # INCREASED to 15 operations to cover ultimate pipeline diversity
        shuffle_order = random.sample(range(15), 15)

        for i in shuffle_order:
            if random.random() < op_p:
                op_strength = self.get_strength()
                
                if i == 0: img = add_smoothing2(img, strength=op_strength) 
                elif i == 1: img = add_motion_blur(img, strength=op_strength) 
                elif i == 2:
                    r = random.random()
                    if r < 0.33: img = add_speckle_noise(img, strength=op_strength)
                    elif r < 0.66: img = add_Poisson_noise(img)
                    else: img = add_salt_and_pepper(img, prob=0.1, strength=op_strength)
                elif i == 3: img = add_resize(img, sf=2, strength=op_strength) 
                elif i == 4:
                    if random.random() < 0.2: img = add_grayscale_drop(img)
                    else: img = add_color_cast_and_fade(img, strength=op_strength) 
                elif i == 5:
                    min_noise = 2 if random.random() < 0.2 else 60
                    img = add_Gaussian_noise(img, noise_level1=min_noise, noise_level2=100, strength=op_strength)
                elif i == 6: img = add_JPEG_noise(img) 
                elif i == 7: img = add_chroma_subsampling(img) 
                elif i == 8: img = add_color_banding(img, strength=op_strength) 
                elif i == 9: img = add_enhance_advanced(img, strength=op_strength) 
                elif i == 10: img = add_chromatic_aberration(img, strength=op_strength) # NEW
                elif i == 11: img = add_moire_pattern(img, strength=op_strength) # NEW
                elif i == 12: img = add_glitch_and_shift(img, strength=op_strength) # NEW
                elif i == 13: img = add_vignetting(img, strength=op_strength) # NEW
                elif i == 14: # (Previously 10) Distractor logic
                    if random.random() > self.distractor_p: #skip
                        continue

                    if random.random() < 0.9:
                        color = (random.random(), random.random(), random.random())
                        text_len = random.randint(0, 10)
                        text = "".join(np.random.choice(list(string.printable), text_len, replace=True))

                        img = cv2.putText(
                            img = np.ascontiguousarray(img),
                            text = text, 
                            org = (random.randint(-100, w), random.randint(0, h+100)), 
                            fontFace = np.random.randint(8), 
                            fontScale = random.random() * 8, 
                            color = color, 
                            thickness = random.randint(1, 8),
                            lineType = np.random.randint(3)
                        )
                    else:
                        try:
                            i2 = random.randint(0, len(self.dataset)-1)
                            sample2 = self.dataset[i2]
                            if isinstance(sample2, dict):
                                key = random.choice(list(sample2.keys()))
                                img2 = sample2[key][0]
                            else:
                                img2 = sample2[0]  
                            if isinstance(img2, list):
                                img2 = img2[0]
                            img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
                            if img2.max() > 1 or img2.min() < 0:
                                img2 = img2/2 + 0.5
                        except Exception:
                            continue
                        
                        img2 = np.ascontiguousarray(img2)
                        small_size_x = random.randint(20, 100)
                        small_size_y = int(small_size_x*(random.random()*0.4 + 0.8))
                        img2 = cv2.resize(img2, (small_size_x, small_size_y), interpolation=random.choice([1, 2, 3]))

                        h_now, w_now = img.shape[:2]
                        ph, pw = img2.shape[:2]
                        x, y = random.randint(-pw, w_now), random.randint(-ph, h_now)

                        x0, y0 = max(x, 0), max(y, 0)
                        x1, y1 = min(x + pw, w_now), min(y + ph, h_now)

                        if x1 <= x0 or y1 <= y0: continue

                        sx0, sy0 = x0 - x, y0 - y
                        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

                        patch = img2[sy0:sy1, sx0:sx1, :]
                        if patch.shape[:2] != (y1 - y0, x1 - x0): continue

                        img[y0:y1, x0:x1, :] = patch

        img = cv2.resize(img, (w, h), interpolation=random.choice([1, 2, 3]))
        return img