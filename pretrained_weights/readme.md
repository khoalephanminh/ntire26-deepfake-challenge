# Pretrained Weights

This directory is intended to store the `.pth` model weights required for inference. 

Due to file size limits, the weights are not hosted on GitHub. Please download them from our [Hugging Face repository](https://huggingface.co/lpmkhoa/hcmusaqua-ntire26-weights).

### Download Instructions

From the root directory of this repository, run:

```bash
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_252.pth --local-dir pretrained_weights
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_252_crop.pth --local-dir pretrained_weights
huggingface-cli download lpmkhoa/hcmusaqua-ntire26-weights dinov2_clip.pth --local-dir pretrained_weights

```

**Required files:**

* `dinov2_252.pth`
* `dinov2_252_crop.pth`
* `dinov2_clip.pth`

