# HCMUS-Aqua's Solution for NTIRE 2026 - Robust Deepfake Detection Challenge @ CVPR 2026

🎉 **News: Our paper is accepted at CVPRW 2026!**

This repository contains the inference pipeline and ensemble code for our submission.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data & Weights Preparation](#2-data--weights-preparation)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Output](#4-output)
5. [Acknowledgements](#5-acknowledgements)
6. [Support](#6-support)

## 1. Environment Setup

We recommend using Conda to ensure strict reproducibility. 

**Prerequisites:** * Linux (tested on Ubuntu)
* Python 3.10
* CUDA 12.8 compatible hardware

Create and activate the environment:
```bash
conda create -n hcmusaqua_submission python=3.10 -y
conda activate hcmusaqua_submission
```

**⚠️ IMPORTANT: Do not use `pip install -r requirements.txt`**
We have included a complete `requirements.txt` file in the repository, but doing a standard pip install will fail due to complex C++ compilation orders (e.g., FlashAttention, MMCV) and PyTorch CUDA dependencies. Please treat the `requirements.txt` file strictly as a reference for debugging.

To safely install the exact dependencies required, you **must** use our automated setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```



## 2. Data & Weights Preparation

Before running the evaluation pipeline, ensure your file structure is configured correctly:

1. **Test Dataset:** Place the test images in the `datasets/publictest_data_final` folder.
2. **Model Weights:** We host our pretrained model weights on Hugging Face. You must download them into the `pretrained_weights/` directory.

We recommend using the `huggingface-cli` to download the weights efficiently. Run the following commands:

```bash
pip install -U "huggingface_hub[cli]"

# Download the weights directly into the pretrained_weights folder
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_252.pth --local-dir pretrained_weights
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_252_crop.pth --local-dir pretrained_weights
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_clip.pth --local-dir pretrained_weights
```

**Required files:**

* `dinov2_252.pth`
* `dinov2_252_crop.pth`
* `dinov2_clip.pth`


## 3. Running the Pipeline

We have provided an automated shell script (`run.sh`) that handles preprocessing, JSON generation, inference across all three models, and final ensembling.

### Evaluating the Public Test
For the public test set, **we have already preprocessed the images and created the JSON configuration files**. 

If you wish to save time, you can open `run.sh`, comment out Steps 1 and 2, and skip directly to **[3/4] Running Inference Models...** to just rerun the predictions and create the submission files.

To execute the full pipeline from scratch, run:
```bash
chmod +x run.sh
./run.sh
```

### Evaluating the Private Test
When evaluating on the private test dataset, please open `run.sh` and modify the configuration variables at the top of the script to match the new dataset paths:

```bash
# --- CONFIGURATION ---
GPU_ID=5
INPUT_DATASET="datasets/privatetest_data_final"  # Update this to the private test folder
CROPPED_DATASET="datasets/privatetest_data_final_cropped"  # Update this destination folder
```

**Save path Note:** Also, please modify the **txt paths** correspondingly in the stage **[3/4] Running Inference Models...**
## 4. Output

Upon successful completion, the final ensembled predictions will be generated at:
`submissions/ensemble_public_test/submission.txt` (or your modified path)

## 5. Acknowledgements
Our codebase is inspired by the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) framework. We thank the authors for their contribution to the community.

## 6. Support

If you encounter any issues reproducing the environment, running the scripts, or generating the final submission file, please email us at: **lpmkhoa22@apcs.fitus.edu.vn**


