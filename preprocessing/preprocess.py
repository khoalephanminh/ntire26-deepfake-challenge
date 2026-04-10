# CUDA_VISIBLE_DEVICES=0 python ntire26-deepfake-challenge/preprocessing/preprocess.py

import torch
import random
import cv2
import os
import numpy as np
import logging
from tqdm import tqdm
import hashlib # Add this for per-image seeding

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Force 1 thread for OpenCV to stop CPU race conditions
cv2.setNumThreads(1) 
torch.use_deterministic_algorithms(True, warn_only=True)

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Force deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sys
import torchvision.transforms.functional as TF
# Trick để lừa basicsr rằng module cũ vẫn tồn tại
sys.modules['torchvision.transforms.functional_tensor'] = TF


from retinaface import RetinaFace
from gfpgan import GFPGANer

import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Preprocess and crop faces from a directory of images.")

parser.add_argument(
    "--input_dir", 
    type=str, 
    default="/raid/dtle/ntire26-deepfake-challenge/datasets/publictest_data_final", 
    help="Path to the directory containing the original images."
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    default="/raid/dtle/ntire26-deepfake-challenge/datasets/publictest_data_final_cropped_original", 
    help="Full path to the output directory where cropped images will be saved."
)

# Parse the arguments
args = parser.parse_args()

# Assign to your variables
INPUT_DIR = args.input_dir
OUT_ORIGINAL_CROP = args.output_dir

# Automatically extract the parent directory for your logging setup
save_parent_dir = os.path.dirname(OUT_ORIGINAL_CROP)

MARGIN_PADDING = 1.3
IS_REFLECT = False
LOG_FILENAME = f"preprocessing_log_{os.path.basename(OUT_ORIGINAL_CROP)}.txt"
os.makedirs(save_parent_dir, exist_ok=True)



print("Loading GFPGAN model...")
gfpganer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)
print("GFPGAN model loaded successfully!")



# Setup Logging to write to a file instead of the console
logging.basicConfig(
    filename=os.path.join(save_parent_dir, LOG_FILENAME),
    filemode='w', # a = append
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def detect_faces_with_fallbacks(img):
    """
    Attempts to detect faces using 9 different strategies.
    Returns: (detections_dict, successful_clean_image, strategy_name)
    """
    # Strategy 0: Raw Image
    detections = RetinaFace.detect_faces(img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, img.copy(), "0: Raw Image"

    # Strategy 1: Bilateral Filter
    clean_img = cv2.bilateralFilter(img, 9, 75, 75)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "1: Bilateral Filter"

    # Strategy 6: Heavy Median Blur
    clean_img = cv2.medianBlur(img, 9)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "6: Heavy Median Blur"

    # Strategy 10: GFPGAN Enhancement
    try:
        _, _, restored_img = gfpganer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True, 
            weight=0.5
        )
        if restored_img is not None:
            detections = RetinaFace.detect_faces(restored_img, threshold=0.9)
            if isinstance(detections, dict) and len(detections) > 0:
                return detections, restored_img, "10: GFPGAN Enhancement"
    except Exception as e:
        #logging.error(f"GFPGAN enhancement failed: {e}")
        print(f"GFPGAN enhancement failed: {e}")

    # Strategy 11: GFPGAN Enhancement + Bilateral Filter
    clean_img = cv2.bilateralFilter(img, 9, 75, 75)
    try:
        _, _, restored_img = gfpganer.enhance(
            clean_img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True, 
            weight=0.5
        )
        if restored_img is not None:
            detections = RetinaFace.detect_faces(restored_img, threshold=0.9)
            if isinstance(detections, dict) and len(detections) > 0:
                return detections, restored_img, "11: GFPGAN Enhancement + Bilateral Filter"
    except Exception as e:
        #logging.error(f"GFPGAN enhancement failed: {e}")
        print(f"GFPGAN enhancement failed: {e}")

    # Strategy 12: GFPGAN Enhancement + Heavy Median Blur
    clean_img = cv2.medianBlur(img, 9)
    try:
        _, _, restored_img = gfpganer.enhance(
            clean_img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True, 
            weight=0.5
        )
        if restored_img is not None:
            detections = RetinaFace.detect_faces(restored_img, threshold=0.9)
            if isinstance(detections, dict) and len(detections) > 0:
                return detections, restored_img, "12: GFPGAN Enhancement + Heavy Median Blur"
    except Exception as e:
        #logging.error(f"GFPGAN enhancement failed: {e}")
        print(f"GFPGAN enhancement failed: {e}")

    # Strategy 2: Sharpening + CLAHE
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clean_img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "2: Sharpening + CLAHE"

    # Strategy 3: Non-Local Means
    clean_img = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "3: Non-Local Means"

    # Strategy 4: Median Blur
    clean_img = cv2.medianBlur(img, 5)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "4: Median Blur"

    # Strategy 5: NLMeans + CLAHE
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_lowlight = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe_lowlight.apply(l)
    clean_img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "5: NLMeans + CLAHE"

    # Strategy 7: MinMax + Grayscale Equalize
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    clean_img = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) 
    detections = RetinaFace.detect_faces(clean_img, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        return detections, clean_img, "7: MinMax + Grayscale Equalize"

    # Strategy 8: Nuclear Option (Upscale + Gamma + Unsharp)
    gamma = 0.5 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bright_img = cv2.LUT(img, table)
    height, width = bright_img.shape[:2]
    upscaled = cv2.resize(bright_img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), 3.0)
    unsharp = cv2.addWeighted(upscaled, 2.0, gaussian_blur, -1.0, 0)
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clean_img_large = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    
    detections = RetinaFace.detect_faces(clean_img_large, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        scaled_detections = {}
        for k, v in detections.items():
            scaled_v = v.copy()
            scaled_v["facial_area"] = [int(coord / 2) for coord in v["facial_area"]]
            scaled_detections[k] = scaled_v
        clean_img_normal = cv2.resize(clean_img_large, (width, height), interpolation=cv2.INTER_AREA)
        return scaled_detections, clean_img_normal, "8: Nuclear Option"

    # Strategy 9: Padding
    h, w = img.shape[:2]
    pad = int(max(h, w) * 0.2) 
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[128, 128, 128])
    gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    clean_img_padded = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    
    detections = RetinaFace.detect_faces(clean_img_padded, threshold=0.9)
    if isinstance(detections, dict) and len(detections) > 0:
        unpadded_detections = {}
        for k, v in detections.items():
            unpadded_v = v.copy()
            x1, y1, x2, y2 = v["facial_area"]
            unpadded_v["facial_area"] = [x1 - pad, y1 - pad, x2 - pad, y2 - pad]
            unpadded_detections[k] = unpadded_v
        # Remove padding from clean image so dimensions match the original
        clean_img_unpadded = clean_img_padded[pad:pad+h, pad:pad+w]
        return unpadded_detections, clean_img_unpadded, "9: Padded + Threshold 0.1"
    
    return {}, img.copy(), "Failed all strategies"

def crop_face_with_margin(img, bbox, margin=MARGIN_PADDING, target_size=(256, 256)):
    """
    Expands the bounding box by the margin parameter, ensures it is a square, 
    pads the image if the box falls out of bounds, and resizes to target_size.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    side_length = max(w, h) * margin
    
    new_x1 = int(cx - side_length / 2)
    new_y1 = int(cy - side_length / 2)
    new_x2 = int(cx + side_length / 2)
    new_y2 = int(cy + side_length / 2)
    
    img_h, img_w = img.shape[:2]
    
    pad_top = max(0, -new_y1)
    pad_bottom = max(0, new_y2 - img_h)
    pad_left = max(0, -new_x1)
    pad_right = max(0, new_x2 - img_w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if IS_REFLECT:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_REFLECT_101)
        else:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        new_x1 += pad_left
        new_x2 += pad_left
        new_y1 += pad_top
        new_y2 += pad_top
        
    cropped = img[new_y1:new_y2, new_x1:new_x2]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

def process_and_crop(input_dir, out_orig_dir):
    os.makedirs(out_orig_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # Get the list of valid images
    # image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"No valid images found in {input_dir}")
        return

    # Trackers for statistics
    success_count = 0
    failed_images = []

    logging.info(f"--- STARTING BATCH PROCESSING: {total_images} images ---")

    # Wrap the loop in tqdm for a console progress bar
    for filename in tqdm(image_files, desc="Processing Images", unit="img"):
        # --- ADD THIS BLOCK ---
        # Generate a unique, deterministic seed for THIS specific file
        file_seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2**32)
        random.seed(file_seed)
        np.random.seed(file_seed)
        torch.manual_seed(file_seed)
        torch.cuda.manual_seed_all(file_seed)
        # ----------------------
        
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            logging.error(f"[{filename}] Could not read image file. Skipping.")
            failed_images.append(filename)
            continue

        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h

        # 1. Run detection
        detections, clean_img, strategy = detect_faces_with_fallbacks(img)

        # 2. Track stats and crop
        if isinstance(detections, dict) and len(detections) > 0:
            success_count += 1
            logging.info(f"[{filename}] SUCCESS - Detected via {strategy}")

            # --- SẮP XẾP FACE: ƯU TIÊN SCORE LỆCH >= 0.2, SAU ĐÓ TỚI CENTER/DIỆN TÍCH ---
            if len(detections) > 1:
                # 0. Sắp xếp các face theo score giảm dần để lấy top 1 và top 2
                # sorted_by_score = sorted(detections.items(), key=lambda item: item[1].get("score", 0), reverse=True)
                # Added item[0] to the lambda tuple
                sorted_by_score = sorted(detections.items(), key=lambda item: (item[1].get("score", 0), item[0]), reverse=True)
                top1_key, top1_info = sorted_by_score[0]
                top2_key, top2_info = sorted_by_score[1]
                
                score_diff = top1_info.get("score", 0) - top2_info.get("score", 0)
                
                # Nếu score lệch nhau từ 0.1 trở lên, chốt luôn face top 1
                if score_diff >= 0.02:
                    best_face_key = top1_key
                else:
                    # Nếu score bám sát nhau, dùng logic vị trí (chứa center / gần center)
                    img_h, img_w = img.shape[:2]
                    img_cx, img_cy = img_w / 2.0, img_h / 2.0
                    
                    faces_containing_center = {} # Sẽ lưu {key: area}
                    all_distances = {}           # Sẽ lưu {key: distance}
                    
                    for k, v in detections.items():
                        x1, y1, x2, y2 = v["facial_area"]
                        face_cx = (x1 + x2) / 2.0
                        face_cy = (y1 + y2) / 2.0
                        
                        # 1. Tính khoảng cách từ tâm face đến tâm ảnh
                        dist = (face_cx - img_cx)**2 + (face_cy - img_cy)**2
                        all_distances[k] = dist
                        
                        # 2. Tính diện tích của bounding box
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Nếu box chứa tâm ảnh, lưu lại diện tích để so sánh
                        if x1 <= img_cx <= x2 and y1 <= img_cy <= y2:
                            faces_containing_center[k] = area
                            
                    if faces_containing_center:
                        # Chọn face CÓ CHỨA center và có DIỆN TÍCH lớn nhất
                        # best_face_key = max(faces_containing_center, key=faces_containing_center.get)
                        best_face_key = max(faces_containing_center, key=lambda k: (faces_containing_center[k], k))
                    else:
                        # Nếu không có face nào chứa center, chọn face GẦN center nhất
                        best_face_key = min(all_distances, key=all_distances.get)
                        
                # Tạo dict mới, đưa best_face lên đầu tiên
                sorted_detections = {best_face_key: detections[best_face_key]}
                for k, v in detections.items():
                    if k != best_face_key:
                        sorted_detections[k] = v
                detections = sorted_detections
            # ----------------------------------------------------------------
            
            for i, (face_key, face_info) in enumerate(detections.items()):
                bbox = face_info["facial_area"]
                x1, y1, x2, y2 = bbox
                face_area = (x2 - x1) * (y2 - y1)
                
                cropped_orig  = crop_face_with_margin(img.copy(),       bbox, margin=MARGIN_PADDING)
                
                name, ext = os.path.splitext(filename)
                if i > 0:
                    save_name = f"{name}_face{i}{ext}" if len(detections) > 1 else filename
                else:
                    save_name = f"{name}{ext}"
                
                # --- KIỂM TRA TỶ LỆ DIỆN TÍCH TRƯỚC KHI CROP ---
                # Nếu diện tích bounding box < 1/9 diện tích ảnh gốc, giữ nguyên ảnh gốc
                if face_area < (img_area / 9.0):
                    logging.info(f"[{filename} - Face {i}] Face area < 1/9 image area. Keeping original image.")
                    cv2.imwrite(os.path.join(out_orig_dir,  save_name), img)
                else:
                    # Proceed with cropping
                    cropped_orig  = crop_face_with_margin(img.copy(),       bbox, margin=MARGIN_PADDING)
                    
                    cv2.imwrite(os.path.join(out_orig_dir, save_name), cropped_orig)
                # ------------------------------------------------
        else:
            # --- REPLACE YOUR EXISTING ELSE BLOCK WITH THIS ---
            failed_images.append(filename)
            logging.warning(f"[{filename}] FAILED - No faces detected. ({strategy}). Copying original.")
            
            # Copy the completely untouched, uncropped image to both output folders
            cv2.imwrite(os.path.join(out_orig_dir, filename), img)
            # ------------------------------------------------

    # --- FINAL STATISTICS LOGGING ---
    fail_count = len(failed_images)
    fail_rate = (fail_count / total_images) * 100

    logging.info("=========================================")
    logging.info("           FINAL STATISTICS              ")
    logging.info("=========================================")
    logging.info(f"Total Images Processed : {total_images}")
    logging.info(f"Successfully Cropped   : {success_count}")
    logging.info(f"Failed to Detect       : {fail_count}")
    logging.info(f"Failure Rate           : {fail_rate:.2f}%")

    print("=========================================")
    print("           FINAL STATISTICS              ")
    print("=========================================")
    print(f"Total Images Processed : {total_images}")
    print(f"Successfully Cropped   : {success_count}")
    print(f"Failed to Detect       : {fail_count}")
    print(f"Failure Rate           : {fail_rate:.2f}%")

    if failed_images:
        logging.info("--- List of Failed Images ---")
        for f in sorted(failed_images):
            logging.info(f"  - {f}")
            
    logging.info("=========================================\n")

if __name__ == "__main__":
    print(f"Starting pipeline. Check '{save_parent_dir}/{LOG_FILENAME}' for details.")
    process_and_crop(INPUT_DIR, OUT_ORIGINAL_CROP)
    print(f"Processing complete! See '{save_parent_dir}/{LOG_FILENAME}' for final statistics.")

