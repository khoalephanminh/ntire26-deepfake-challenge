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

# from trainer.trainer import Trainer
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

    def normalize_custom(self, img, mean, std):
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    @staticmethod
    def collate_fn(batch):
        """
        batch là list của các dict được trả về bởi __getitem__:
          [
            { "image": [1 ảnh res1], "image_2": [6 ảnh res2]?, "image_3": [1 ảnh res3], "label": lbl0 },
            { "image": [1 ảnh res1], "image_2": [6 ảnh res2]?, "image_3": [1 ảnh res3], "label": lbl1 },
            ...
          ]

        Chúng ta sẽ gom mỗi độ phân giải riêng thành tensor shape (B, 6, C, H, W). 
        Trả về dict:
          {
            "image":   torch.Tensor of shape (B, C, H, W),
            "image_2": torch.Tensor of shape (B, 6, C, H2, W2)   (nếu có),
            "image_3": torch.Tensor of shape (B, C, H3, W3)   (nếu có),
            "label":   torch.LongTensor of shape (B,)
          }
        """


        # -- Xử lý resolution 1 (bắt buộc phải có)
        # batch_images_res1 là list length=B, mỗi phần tử lại là list length=6 (6 crops)
        batch_images_res0 = [d["image_original"] for d in batch]
        batch_images_res1 = [d["image_1"] for d in batch]
        batch_images_res2 = [d["image_2"] for d in batch]  # mỗi phần tử list length=6
        batch_images_res3 = [d["image_3"] for d in batch]

        # Đưa thành tensor (B, 6, C, H, W)
        # Mỗi batch_images_res1[i] là list(6) tensor shape (C, H, W)

        # tensor_res0 = torch.stack(
        #     [torch.stack(img_list, dim=0) for img_list in batch_images_res0],
        #     dim=0
        # )
        tensor_res1 = torch.stack(
            [torch.stack(img_list, dim=0) for img_list in batch_images_res1],
            dim=0
        )
        tensor_res2 = torch.stack(
            [torch.stack(img_list, dim=0) for img_list in batch_images_res2],
            dim=0
        )
        tensor_res3 = torch.stack(
            [torch.stack(img_list, dim=0) for img_list in batch_images_res3],
            dim=0
        )
        
        out = {
            "image_original": batch_images_res0,  # (B, 1, C, H, W)
            "image_1": tensor_res1,  # (B, 1, C, H, W)
            "image_2": tensor_res2,  # (B, 6, C, H2, W2)
            "image_3": tensor_res3  # (B, 1, C, H3, W3)
        }
        # out['landmark'] = None
        # out['mask'] = None

        return out

    def five_crops_tensor(self, img_tensor: torch.Tensor, sixImg: bool = False, origin: bool = False):
        """
        Nhận vào img_tensor có shape (C, H, W) – đã normalize và transform xong.
        Trả về list 6 tensor, mỗi cái shape (C, H, W):
          0) origin (chính img_tensor)
          1) crop top-left (2/3 x 2/3 → resize về HxW)
          2) crop top-right (2/3 x 2/3 → resize về HxW)
          3) crop bottom-left (2/3 x 2/3 → resize về HxW)
          4) crop bottom-right (2/3 x 2/3 → resize về HxW)
          5) crop center (2/3 x 2/3 → resize về HxW)
        Nếu sixImg = False thì không crop, trả về [img_tensor]
        """
        if not sixImg:
            return [img_tensor]

        # Nếu img_tensor có kèm batch dim (1, C, H, W), ta squeeze:
        squeezed = False
        if img_tensor.dim() == 4 and img_tensor.size(0) == 1:
            img_tensor = img_tensor.squeeze(0)
            squeezed = True

        # img_tensor giờ phải là (C, H, W)
        C, H, W = img_tensor.shape

        # Kích thước crop = 2/3 × H và 2/3 × W
        crop_h = int(round(H * 2 / 3))
        crop_w = int(round(W * 2 / 3))

        # Tính tọa độ 5 điểm crop
        coords = [
            (0, 0),                                 # top-left
            (W - crop_w, 0),                        # top-right
            (0, H - crop_h),                        # bottom-left
            (W - crop_w, H - crop_h),               # bottom-right
            ((W - crop_w) // 2, (H - crop_h) // 2)  # center
        ]

        crops = []
        # 0) Thêm origin (giữ nguyên)
        if origin is True:
            crops.append(img_tensor.clone())

        # 1–5) Các crop phía trên, rồi resize về (H, W) bằng bilinear
        for (x0, y0) in coords:
            patch = img_tensor[:, y0 : y0 + crop_h, x0 : x0 + crop_w]  # (C, crop_h, crop_w)
            patch = patch.unsqueeze(0)  # (1, C, crop_h, crop_w)
            resized = F.interpolate(patch, size=(H, W), mode='bilinear', align_corners=False)  # (1, C, H, W)
            resized = resized.squeeze(0)  # (C, H, W)
            crops.append(resized)

        # Nếu ban đầu có batch dim, thêm lại:
        if squeezed:
            crops = [p.unsqueeze(0) for p in crops]

        return crops  # list length = 6
        
    def load_rgb(self, file_path):
        size = self.config['resolution'] # if self.mode == "train" else self.self.config['resolution']
        size_2 = self.config['resolution_2']
        size_3 = self.config['resolution_3']
        size_3 = self.config['resolution_3']

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_2 = img.copy()
        img_3 = img.copy()

        if size_2 < size:
            img_2 = cv2.resize(img_2, (size_2, size_2), interpolation=cv2.INTER_AREA)
        elif size_2 > size:
            img_2 = cv2.resize(img_2, (size_2, size_2), interpolation=cv2.INTER_CUBIC)
        image_2 = Image.fromarray(np.array(img_2, dtype=np.uint8))
        image_2 = np.array(image_2)  # Convert to numpy array for data augmentation

        if size_3 < size:
            img_3 = cv2.resize(img_3, (size_3, size_3), interpolation=cv2.INTER_AREA)
        elif size_3 > size:
            img_3 = cv2.resize(img_3, (size_3, size_3), interpolation=cv2.INTER_CUBIC)
        image_3 = Image.fromarray(np.array(img_3, dtype=np.uint8))
        image_3 = np.array(image_3)

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(np.array(img, dtype=np.uint8))

        image = np.array(image)  # Convert to numpy array for data augmentation

        image0 = image

            
        image = self.normalize(self.to_tensor(image))
        image_2 = self.normalize(self.to_tensor(image_2))
        image_3 = self.normalize_custom(self.to_tensor(image_3), self.config['mean_3'], self.config['std_3'])
        return [image0, image, image_2, image_3]  # Return the original image and normalized image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensors = self.load_rgb(image_path)

        # 2) Xác định xem image_tensors ở dạng single tensor hay list (nhiều resolution)
        if isinstance(image_tensors, list):
            # Giả sử thứ tự là [img_res1, img_res2, img_res3]
            img_res1 = image_tensors[1]
            img_res2 = image_tensors[2] if len(image_tensors) >= 2 else None
            img_res3 = image_tensors[3] if len(image_tensors) >= 3 else None
        else:
            img_res1 = image_tensors
            img_res2 = None
            img_res3 = None

        # 3) Tạo 6 crops cho từng resolution
        crops_res1 = self.five_crops_tensor(img_res1)  # list length=6
        crops_res2 = self.five_crops_tensor(img_res2, sixImg = True) if img_res2 is not None else None # Effb4
        crops_res3 = self.five_crops_tensor(img_res3, sixImg = True) if img_res3 is not None else None # CLIP
        
        out = {
            "image_original": image_tensors[0],
            "image_1": crops_res1,    # luôn có 6 ảnh (C, H, W)
            "image_2": crops_res2,
            "image_3": crops_res3
        }
            
        # return [image_tensors[0], crops_res1, crops_res2, crops_res3]
        return out

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=8)
    
    return dataloader

def visualize_gradcam(model, input_tensors, output_folder, item_idx, use_smooth=0):
    model_name = type(model.backbone).__name__
    print("modelname=", model_name)
    # return

    # Ensure the model is in evaluation mode
    os.makedirs(output_folder, exist_ok=True)
    model.eval()


    # Find the last convolutional layer        

    # process batch -> input images also batch of 6 -> target layer is the same


    grad_model = model.backbone

    if model_name == 'MixModel_x3_crop5img':
        layer_model_1 = model.backbone.model_2.efficientnet._conv_head
        target_layers_1 = [layer_model_1]

        for param in model.backbone.model_1.parameters():
            param.requires_grad = True
        for param in model.backbone.model_2.parameters():
            param.requires_grad = True
        for param in model.backbone.model_3.parameters():
            param.requires_grad = True

    targets = [ClassifierOutputTarget(1)]

    input_tensor_original = input_tensors["image_original"]
    input_tensor_1 = input_tensors["image_1"].to(device)
    input_tensor_2 = input_tensors["image_2"].to(device)
    input_tensor_3 = input_tensors["image_3"].to(device)

    grad_model.grad_images = [input_tensor_1, input_tensor_2, input_tensor_3]

    print("shapes=", input_tensor_original[0].shape, input_tensor_1.shape, input_tensor_2.shape, input_tensor_3.shape)

    # with GradCAMPlusPlus(model=model.backbone, target_layers=target_layers) as cam:
    with GradCAMPlusPlus(model=grad_model, target_layers=target_layers_1) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        B_outer = input_tensor_2.size(0)    # outer batch size
        B_inner = input_tensor_2.size(1)    # inner batch size (5)

        for outer_idx in range(B_outer):
            torch.set_grad_enabled(True)  # required for grad cam
            # For each sample in outer batch
            inner_batch_tensor = input_tensor_2[outer_idx]   # shape: (5, 3, 380, 380)

            if use_smooth:
                grayscale_cam = cam(input_tensor=inner_batch_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
            else:
                grayscale_cam = cam(input_tensor=inner_batch_tensor, targets=targets)
            
            print("grayscale_cam=", grayscale_cam.shape)
        

            # Convert inner batch to numpy: shape becomes (5, 380, 380, 3)
            inner_batch_numpy = inner_batch_tensor.cpu().numpy().transpose(0, 2, 3, 1)

            # Normalize each image in the inner batch to [0, 1]
            min_vals = inner_batch_numpy.min(axis=(1, 2, 3), keepdims=True)  # shape: (5, 1, 1, 1)
            max_vals = inner_batch_numpy.max(axis=(1, 2, 3), keepdims=True)  # shape: (5, 1, 1, 1)
            normalized_inner_batch = (inner_batch_numpy - min_vals) / (max_vals - min_vals)

            # For grayscale_cam and cam.outputs, assume matching inner batch size (5)
            grayscale_cam_batch = grayscale_cam[outer_idx]      # shape: (5, H, W)
            preds_batch = cam.outputs                # shape: (5, num_classes)
            prob = torch.softmax(preds_batch[outer_idx], dim=1)[:, 1]
            prob = f"{prob.item():.6f}"

            output_path_original = os.path.join(output_folder, 
                                            f'gradcam_outer{outer_idx}_original.png')
            cv2.imwrite(output_path_original, cv2.cvtColor(numpy_image_original, cv2.COLOR_RGB2BGR))
            print(f"Original image saved: {output_path_original}")

            for inner_idx in range(B_inner):
                rgb_img = normalized_inner_batch[inner_idx]  # (380, 380, 3)
                grayscale_cam_single = grayscale_cam_batch[inner_idx]  # (380, 380)

                # Create Grad-CAM visualization
                visualization = show_cam_on_image(rgb_img, grayscale_cam_single, use_rgb=True)

                # Save Grad-CAM visualization
                output_path = os.path.join(output_folder, 
                                        f'gradcam_outer{outer_idx}_inner{inner_idx}_{prob}_mavg.png')
                plt.imsave(output_path, visualization)
                print(f"Grad-CAM saved: {output_path}")

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
        model.load_state_dict(ckpt, strict=False) # It was True, but position_ids
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # Load images in batches
    dataloader = load_images_in_batches(input_folder, config, batch_size)
    
    #remove output_folder if exist
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