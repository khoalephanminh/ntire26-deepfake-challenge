import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from dataset.fsbi_utils import get_dwt

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument('--input_folder', type=str, help='Folder path to images')
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')      
parser.add_argument('--output_folder', type=str, default=None, help='Path to save the results')            
parser.add_argument('--use_smooth', type=int, default=0, help='Whether to use smooth gradcam')  

#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class ImageDataset(Dataset):
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.image_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(input_folder)
                            for file in files if file.endswith('.png') or file.endswith('.jpg')]
        self.config = config

    def __len__(self):
        return len(self.image_paths)

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def load_rgb(self, file_path):
        size = self.config['resolution'] # if self.mode == "train" else self.self.config['resolution']

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(np.array(img, dtype=np.uint8))

        image = np.array(image)  # Convert to numpy array for data augmentation

        image0 = image
        
        print("model name=", self.config['model_name'])
        print("backbone name=", self.config['backbone_name'])

        # if self.config['model_name'] == 'fsbi':
        #     image = get_dwt(image, (self.config['resolution'], self.config['resolution']))

        if self.config['model_name'] == 'sbi_crop':
            print("crop image 0.2")
            image = self.random_crop(image, 0.2)  
            
        image = self.normalize(self.to_tensor(image))
        return [image0, image]  # Return the original image and normalized image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        images = self.load_rgb(image_path)
        return images

    def random_crop(self, image, crop_percent, resolution = (380, 380)):
        """
        Randomly crops an image by a given percentage from an already enlarged image (1.3x the original).

        :param image: Input image (numpy array) assumed to be 1.3x the original size.
        :param crop_percent: Fraction of the original size to keep (e.g., 0.2 means cropping 20%).
        :return: Cropped image.
        """
        H_curr, W_curr = image.shape[:2]  # Get current size

        # Compute new crop size
        scale_factor = (1 + crop_percent) / 1.3  # Example: (1.2 / 1.3) for 20% crop
        H_new, W_new = int(H_curr * scale_factor), int(W_curr * scale_factor)

        # Compute the amount to crop from each side
        crop_top = (H_curr - H_new) // 2
        crop_bottom = H_curr - H_new - crop_top
        crop_left = (W_curr - W_new) // 2
        crop_right = W_curr - W_new - crop_left

        # Perform cropping
        cropped_image = image[crop_top:H_curr - crop_bottom, crop_left:W_curr - crop_right]
        cropped_image = cv2.resize(cropped_image, resolution, interpolation=cv2.INTER_CUBIC) # hard coding 380 for now

        return cropped_image

def load_images_in_batches(input_folder, config, batch_size):
    dataset = ImageDataset(input_folder, config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def visualize_gradcam(model, input_tensors, output_folder, item_idx, use_smooth=0):
    model_name = type(model.backbone).__name__
    print("modelname=", model_name)
    # return

    # Ensure the model is in evaluation mode
    os.makedirs(output_folder, exist_ok=True)
    model.eval()


    # Find the last convolutional layer        
    
    grad_model = model.backbone

    if model_name == 'Xception':
        target_layers = [model.backbone.conv4]  # This works, but not sure the last layer
    if model_name == 'EfficientNetB4':
        # target_layers = [model.backbone.efficientnet._blocks[22]]
        target_layers = [model.backbone.efficientnet._conv_head]
    if model_name == 'CvT':
        target_layers = [model.backbone.cvt.cvt.encoder.stages[2].layers[19].attention.attention.convolution_projection_value.convolution_projection.convolution] # this is correct
        # target_layers = [model.backbone.cvt.cvt.encoder.stages[0].layers[1].attention.attention.convolution_projection_value.convolution_projection.convolution]
    if model_name == 'MixModel' or model_name == 'MixModel_x3':
        layer_model_1 = model.backbone.model_1.cvt.cvt.encoder.stages[2].layers[19].attention.attention.convolution_projection_value.convolution_projection.convolution
        layer_model_2 = model.backbone.model_2.cvt.cvt.encoder.stages[2].layers[19].attention.attention.convolution_projection_value.convolution_projection.convolution
        # layer_model_2 = model.backbone.model_2.efficientnet._conv_head
        # layer_model_3 = model.backbone.model_3.svd_model.vision_model.encoder.layers[-1].self_attn.out_proj

        target_layers_avg = [layer_model_1, layer_model_2]
        target_layers_1 = [layer_model_1]
        target_layers_2 = [layer_model_2]
        # target_layers_3 = [layer_model_3]
        
        # target_layers = [model.backbone.model_2.efficientnet._conv_head]
        # print("target_layers=", target_layers)
        for param in model.backbone.model_1.parameters():
            param.requires_grad = True

    targets = [ClassifierOutputTarget(1)]

    input_tensor_original = input_tensors[0]
    input_tensor = input_tensors[1]
    input_tensor = input_tensor.to(device)
    
    # Construct the CAM object once, and then re-use it on many images.
    # with GradCAMPlusPlus(model=model.backbone, target_layers=target_layers) as cam:
    with GradCAMPlusPlus(model=grad_model, target_layers=target_layers_1) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        torch.set_grad_enabled(True)  # required for grad cam
        if use_smooth:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        else:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        for i in range(input_tensor.size(0)):
            single_image_tensor = input_tensor[i].unsqueeze(0)
            rgb_img = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            grayscale_cam_single = grayscale_cam[i, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_single, use_rgb=True)

            pred = cam.outputs
            print("pred=", pred[0])
            prob = torch.softmax(pred, dim=1)[:, 1]
            print("prob=", prob)

            #round prob to 2 decimal places
            prob = f"{prob.item():.6f}"
            output_path = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_m1.png')
            output_path2 = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_o.png')

            plt.imsave(output_path, visualization)

    with GradCAMPlusPlus(model=grad_model, target_layers=target_layers_2) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        torch.set_grad_enabled(True)  # required for grad cam
        if use_smooth:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        else:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        for i in range(input_tensor.size(0)):
            single_image_tensor = input_tensor[i].unsqueeze(0)
            rgb_img = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            grayscale_cam_single = grayscale_cam[i, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_single, use_rgb=True)

            pred = cam.outputs
            prob = torch.softmax(pred, dim=1)[:, 1]

            #round prob to 2 decimal places
            prob = f"{prob.item():.6f}"
            output_path = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_m2.png')

            plt.imsave(output_path, visualization)

    # with GradCAMPlusPlus(model=grad_model, target_layers=target_layers_3) as cam:
    #     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #     torch.set_grad_enabled(True)  # required for grad cam
    #     if use_smooth:
    #         grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    #     else:
    #         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
    #     for i in range(input_tensor.size(0)):
    #         single_image_tensor = input_tensor[i].unsqueeze(0)
    #         rgb_img = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
    #         rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    #         grayscale_cam_single = grayscale_cam[i, :]
    #         visualization = show_cam_on_image(rgb_img, grayscale_cam_single, use_rgb=True)

    #         pred = cam.outputs
    #         prob = torch.softmax(pred, dim=1)[:, 1]

    #         #round prob to 2 decimal places
    #         prob = f"{prob.item():.6f}"
    #         output_path = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_m3.png')

    #         plt.imsave(output_path, visualization)

    with GradCAMPlusPlus(model=grad_model, target_layers=target_layers_avg) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        torch.set_grad_enabled(True)  # required for grad cam
        if use_smooth:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        else:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        for i in range(input_tensor.size(0)):
            single_image_tensor = input_tensor[i].unsqueeze(0)
            rgb_img = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            grayscale_cam_single = grayscale_cam[i, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam_single, use_rgb=True)

            pred = cam.outputs
            print("pred=", pred[0])
            prob = torch.softmax(pred, dim=1)[:, 1]
            print("prob=", prob)

            #round prob to 2 decimal places
            prob = f"{prob.item():.6f}"
            output_path = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_mavg.png')
            output_path2 = os.path.join(output_folder, f'gradcam_{item_idx}_{prob}_o.png')

            plt.imsave(output_path, visualization)
            
            # print("numpy_image=", input_tensor_original[i].shape)
            # print("numpy_image=", input_tensor_original[i])
            numpy_image = input_tensor_original[i].cpu().numpy()  # Change shape to (H, W, C)
            numpy_image = numpy_image.astype(np.uint8)
            cv2.imwrite(output_path2, cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

            print(f"Grad-CAM visualization saved to {output_path}")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    if args.output_folder:
        output_folder = args.output_folder
    
    if args.input_folder:
        input_folder = args.input_folder

    if args.batch_size:
        batch_size = args.batch_size

    use_smooth = args.use_smooth

    print("batch_size=", batch_size)
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    if config.get('confusion_matrix', False):
        global write_confusion_matrix
        write_confusion_matrix = True
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        ckpt = {k.replace('module.backbone.', 'backbone.'): v for k, v in ckpt.items()} # added for cvt
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # Load images in batches
    dataloader = load_images_in_batches(input_folder, config, batch_size)
    
    # remove output_folder if exist
    if os.path.exists(output_folder):
        os.system(f"rm -r {output_folder}")

    # Process batches through the model
    print("use_smooth=", use_smooth)
    idx = 0
    for batch in dataloader:
        idx += 1
        # print("batch len, type=", len(batch[0]), type(batch[0]))
        # print("unique=", torch.unique(batch[0]))
        visualize_gradcam(model, batch, output_folder, item_idx = idx, use_smooth = use_smooth)

if __name__ == '__main__':
    main()