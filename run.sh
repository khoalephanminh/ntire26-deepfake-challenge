#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

# Force strict environment isolation to prevent ~/.local bleed
export PYTHONNOUSERSITE=1

# --- CONFIGURATION ---
# Organizers can change this GPU ID to match their hardware
GPU_ID=5
INPUT_DATASET="datasets/publictest_data_final"
CROPPED_DATASET="datasets/publictest_data_final_cropped_original"

echo "=========================================="
echo " Starting NTIRE Deepfake Evaluation Pipeline"
echo "=========================================="

echo "[1/4] Preprocessing Dataset..."
CUDA_VISIBLE_DEVICES=$GPU_ID python preprocessing/preprocess.py \
    --input_dir "$INPUT_DATASET" \
    --output_dir "$CROPPED_DATASET"

echo "[2/4] Building JSON Configuration Files..."
python preprocessing/build_ntire_test_json.py \
    --image_dir "$INPUT_DATASET" \
    --output preprocessing/dataset_json/publictest_data_final.json

python preprocessing/build_ntire_test_json.py \
    --image_dir "$CROPPED_DATASET" \
    --output preprocessing/dataset_json/publictest_data_final_cropped_original.json

echo "[3/4] Running Inference Models..."
# Model 1: DINOv2
echo " -> Running DINOv2 (Original Data)..."
CUDA_VISIBLE_DEVICES=$GPU_ID python training/inference.py \
    --detector_path ./training/config/detector/dinov2_252.yaml \
    --test_dataset "publictest_data_final" \
    --weights_path pretrained_weights/dinov2_252.pth \
    --output_file submissions/dinov2_252_public_test/submission.txt

# Model 2: DINOv2 on Cropped Data
echo " -> Running DINOv2 (Cropped Data)..."
CUDA_VISIBLE_DEVICES=$GPU_ID python training/inference.py \
    --detector_path ./training/config/detector/dinov2_252.yaml \
    --test_dataset "publictest_data_final_cropped_original" \
    --weights_path pretrained_weights/dinov2_252_crop.pth \
    --output_file submissions/dinov2_252_crop_public_test/submission.txt

# Model 3: DINOv2 + CLIP
echo " -> Running DINOv2 + CLIP..."
CUDA_VISIBLE_DEVICES=$GPU_ID python training/inference.py \
    --detector_path ./training/config/detector/dinov2_clip_ft_phase2.yaml \
    --test_dataset "publictest_data_final" \
    --weights_path pretrained_weights/dinov2_clip.pth \
    --output_file submissions/dinov2_clip_public_test/submission.txt

echo "[4/4] Ensembling Submissions..."
python training/ensemble_submissions.py \
  --sub_files submissions/dinov2_252_public_test/submission.txt \
              submissions/dinov2_252_crop_public_test/submission.txt \
              submissions/dinov2_clip_public_test/submission.txt \
  --weights 2 1 2 \
  --output_file submissions/ensemble_public_test/submission.txt 

echo "=========================================="
echo " Pipeline Complete! Final submission saved to:"
echo " submissions/ensemble_public_test/submission.txt"
echo "=========================================="