'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
'''

import os
import argparse
import logging
import numpy as np
import math
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from metrics.registry import BACKBONE

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from detectors import DETECTOR

logger = logging.getLogger(__name__)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        # used for iid
        final_channel = 512
        if self.mode == 'adjust_channel_iid':
            final_channel = 512
            self.mode = 'adjust_channel'
        self.last_linear = nn.Linear(final_channel, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(final_channel, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
           
    def fea_part1_0(self, x):
        print("fea_part1_0")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  

        return x

    def fea_part1_1(self, x):  
        print("fea_part1_1")
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part1(self, x):
        print("fea_part1")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)     
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part2(self, x):
        print("fea_part2")
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        print("fea_part3")
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        print("fea_part4")
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        print("fea_part5")
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x
     
    def features(self, input, show_gradcam=True):
        print("features")
        x = self.fea_part1(input)    

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)

        # if show_gradcam:
        #     self.visualize_gradcam(input)
        
        return x

    def classifier(self, features,id_feat=None):
        print("classifier")
        # for iid
        if self.mode == 'adjust_channel':
            x = features
        else:
            x = self.relu(features)

        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        self.last_emb = x
        # for iid
        if id_feat!=None:
            out = self.last_linear(x-id_feat)
        else:
            out = self.last_linear(x)
        print("end classifier")
        return out

    def forward(self, input, show_gradcam=True):
        print("forward")
        x = self.features(input)
        out = self.classifier(x)

        if show_gradcam:
            return out
        else:  
            return out, x

    def find_last_conv_layer(self):
        # Iterate through the layers in reverse order to find the last convolutional layer
        for layer in reversed(list(self.modules())):
            if isinstance(layer, (nn.Conv2d, SeparableConv2d)):
                return layer
        raise ValueError("No convolutional layer found in the model")

    def visualize_gradcam(self, input_tensor, raw_id):
        print("visu")
        # Ensure the model is in evaluation mode
        output_folder = './gradcam_xception'
        os.makedirs(output_folder, exist_ok=True)
        self.eval()

        # Find the last convolutional layer
        target_layers = [self.find_last_conv_layer()]
        print("target_layers", target_layers)
        targets = [ClassifierOutputTarget(1)]
        targets = None
        
        # Construct the CAM object once, and then re-use it on many images.
        with GradCAM(model=self, target_layers=target_layers) as cam:
            print("i'm 1")
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            torch.set_grad_enabled(True) # required for grad cam
            # cam = self._initialize_grad_cam(False)
            # for param in self.parameters():
            #     param.requires_grad = True

            print("i'm 2")
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            print("i'm 3")
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            print("i'm 4")
            # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            model_outputs = cam.outputs
            print("i'm 5")
            print("model_outputs", model_outputs)

        # # Iterate over the batch and save each Grad-CAM visualization
        # for i in range(input_tensor.size(0)):
        #     # Extract the single image tensor and add a batch dimension
        #     single_image_tensor = input_tensor[i].unsqueeze(0)

        #     # Generate Grad-CAM
        #     grayscale_cam = cam(input_tensor=single_image_tensor, targets=None)

        #     # Convert grayscale CAM to heatmap
        #     input_image = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
        #     input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        #     visualization = show_cam_on_image(input_image, grayscale_cam[0], use_rgb=True)

        #     # Save the Grad-CAM heatmap to the output folder
        #     output_path = os.path.join(output_folder, f'gradcam_{raw_id}_{i}.png')
        #     plt.imsave(output_path, visualization)
        #     print(f"Grad-CAM visualization saved to {output_path}")

model = Xception(xception_config={"num_classes": 2, "mode": "adjust_channel", "inc": 3, "dropout": 0.5})
model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_class = DETECTOR['xception']
# # model = model_class(config).to(device)
# print("model_class=", model_class)
target_layers = [model.conv4]

# Create a random input tensor image for your model
input_tensor = torch.randn(1, 3, 256, 256)

# We have to specify the target we want to generate the CAM for.
targets = [ClassifierOutputTarget(1)]
# targets = None

# Create a random RGB image for visualization
rgb_img = np.random.rand(256, 256, 3)
# Ensure the values are in the range [0, 1]
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Construct the CAM object once, and then re-use it on many images.
with torch.no_grad():
    with GradCAM(model=model, target_layers=target_layers) as cam:
        torch.set_grad_enabled(True) # required for grad cam
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