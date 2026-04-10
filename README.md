# NTIRE Deepfake Detection Challenge Submission

This repository contains the inference pipeline and ensemble code for our submission.

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
2. **Model Weights:** Put our pretrained model weights and place them inside the `pretrained_weights/` directory (we already put it in our zip file):
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

## 5. Support

If you encounter any issues reproducing the environment, running the scripts, or generating the final submission file, please email us at: **lpmkhoa22@apcs.fitus.edu.vn**


