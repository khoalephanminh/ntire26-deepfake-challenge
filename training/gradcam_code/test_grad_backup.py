import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import os
import numpy as np

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]

# Create a random input tensor image for your model
input_tensor = torch.randn(1, 3, 224, 224)

# We have to specify the target we want to generate the CAM for.
targets = [ClassifierOutputTarget(281)]

# Create a random RGB image for visualization
rgb_img = np.random.rand(224, 224, 3)
# Ensure the values are in the range [0, 1]
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Construct the CAM object once, and then re-use it on many images.
with GradCAM(model=model, target_layers=target_layers) as cam:
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # You can also get the model outputs without having to redo inference
    model_outputs = cam.outputs

    # Save the visualization to a folder
    output_folder = 'cam_images'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'cam_image.png')
    plt.imsave(output_path, visualization)