#!/bin/bash
# Exit immediately if any command fails
set -e 

# Force pip to ignore ~/.local and install directly into Conda
export PYTHONNOUSERSITE=1

echo "=========================================="
echo " Starting NTIRE Environment Setup"
echo "=========================================="

echo "[1/5] Installing Core Deep Learning Anchor..."
# Installing this first guarantees no other package can downgrade your CUDA build
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

echo "[2/5] Fixing Build Tools & Compilers..."
# setuptools, wheel, and cmake are required to successfully build C++ packages like dlib and mmcv
pip install setuptools wheel cmake packaging==24.0 ninja

echo "[3/5] Installing Heavy/Tricky CUDA Extensions..."
# FlashAttention must bypass build isolation to "see" PyTorch
# pip install flash_attn==2.8.0.post2 --no-build-isolation
# # OpenMIM safely downloads pre-compiled MMCV without failing on C++ compilation
# pip install openmim
# mim install mmcv-full==1.7.2

echo "[4/5] Installing Main Vision & Audio Libraries..."
# This includes all your primary models, utilities, and the newly added GFPGAN/PyWavelets
pip install \
    albumentations dlib efficientnet_pytorch einops fvcore grad-cam \
    imageio imgaug kornia lmdb loralib matplotlib mpi4py "numpy<2.0.0" \
    opencv-python-headless peft Pillow psutil PyYAML retina-face \
    scikit-image scikit-learn scipy simplejson timm tqdm \
    "transformers[vision]" accelerate yacs h5py tensorboard torchio \
    SimpleITK librosa soundfile cffi \
    PyWavelets gfpgan tf-keras basicsr facexlib \
    mpmath sympy ftfy pyjwt regex

echo "[5/5] Installing OpenAI CLIP Safely..."
# --no-deps prevents GitHub scripts from secretly breaking your environment
pip install git+https://github.com/openai/CLIP.git@ded190a052fdf4585bd685cee5bc96e0310d2c93 --no-deps

echo "=========================================="
echo " Environment setup completed successfully!"
echo "=========================================="