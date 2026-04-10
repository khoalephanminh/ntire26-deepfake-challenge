"""
eval pretained model.
"""
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument('--folder_path', type=str, help='Folder path to images')
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')                    

#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

write_confusion_matrix = False

def load_images_in_batches(folder_path, config, batch_size):
    images = []
    batch_images = []
    
    # Iterate recursively every image in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image = load_rgb(os.path.join(root, file), config)
                batch_images.append(image)
                
                # If batch size is reached, process the batch
                if len(batch_images) == batch_size:
                    images.append(torch.stack(batch_images))
                    batch_images = []
    
    # Process any remaining images in the last batch
    if batch_images:
        images.append(torch.stack(batch_images))
    
    return images

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def to_tensor(img):
    return T.ToTensor()(img)

def normalize(img, config):
    """
    Normalize an image.
    """
    mean = config['mean']
    std = config['std']
    normalize = T.Normalize(mean=mean, std=std)
    return normalize(img)


def load_rgb(file_path, config):
    """
    Load an RGB image from a file path and resize it to a specified resolution.

    Args:
        file_path: A string indicating the path to the image file.

    Returns:
        An Image object containing the loaded and resized image.

    Raises:
        ValueError: If the loaded image is None.
    """
    size = config['resolution'] # if self.mode == "train" else self.config['resolution']

    img = cv2.imread(file_path)
    if img is None:
        raise ValueError('Loaded image is None: {}'.format(file_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(np.array(img, dtype=np.uint8))

    image = np.array(image)  # Convert to numpy array for data augmentation
    image = normalize(to_tensor(image), config)
    return image

def visualize_gradcam(model, input_tensor, raw_id=None):
    # Ensure the model is in evaluation mode
    output_folder = './gradcam_test_0'
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    # Find the last convolutional layer        
    target_layers = [model.backbone.conv4] # This works, but not sure the last layer
    targets = [ClassifierOutputTarget(1)]
    input_tensor = input_tensor.to(device)
    
    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model.backbone, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        torch.set_grad_enabled(True) # required for grad cam
        print("input_tensor=", input_tensor.shape)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        single_image_tensor = input_tensor[0].unsqueeze(0)
        rgb_img = single_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        pred = cam.outputs
        print("pred=", pred[0])
        prob = torch.softmax(pred[0], dim=1)[:, 1]
        print("prob=", prob)

        if raw_id is None:
            raw_id = random.randint(0, 10000)
        output_path = os.path.join(output_folder, f'gradcam_{prob.item()}.png')
        plt.imsave(output_path, visualization)
        print(f"Grad-CAM visualization saved to {output_path}")


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    if args.folder_path:
        folder_path = args.folder_path

    if args.batch_size:
        batch_size = args.batch_size
    
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
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # print("model=", model)
    
    image = load_rgb('/raid/dtle/deepfake/DeepfakeBench/datasets/rgb/Celeb-DF-v2/Celeb-synthesis/frames/id0_id2_0002/011.png', config)
    print("image=: ", image.shape)
    # start testing
    images = []
    #iterate recursively every image in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image = load_rgb(os.path.join(root, file), config)
                images.append(image)

    batch_size = 32  # Define your batch size
    images = load_images_in_batches(folder_path, config, batch_size)
    print("images=", images.shape)
    visualize_gradcam(model, images)

if __name__ == '__main__':
    main()
