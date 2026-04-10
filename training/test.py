"""
eval pretrained model.
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

from dataset.abstract_dataset_83 import DeepfakeAbstractBaseDataset_83
from dataset import CropImgDataset
# from dataset.ff_blend import FFBlendDataset
# from dataset.fwa_blend import FWABlendDataset
# from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='path_to_detector_yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='path_to_weights')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

write_confusion_matrix = False


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset

        if 'dataset_type' in config and config['dataset_type'] == 'crop_img':
            test_set = CropImgDataset(config=config, mode='test')
        
        else:
            test_set = DeepfakeAbstractBaseDataset_83(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        data_dict['index'] = i
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if 'image_2' in data_dict:
            data_dict['image_2'] = data_dict['image_2'].to(device)
        if 'image_3' in data_dict:
            data_dict['image_3'] = data_dict['image_3'].to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            if k != 'label' and k != 'pred':
                tqdm.write(f"{k}: {v}")
        
        is_hoang = False # or True
        if is_hoang:
            # =================================================================
            # THÊM MỚI: GHI KẾT QUẢ RA FILE TXT VỚI TAB ALIGNMENT
            # =================================================================
            output_txt_path = '/raid/dtle/NTIRE26-DeepfakeDetection/misc/tmp-hoang/output.txt'
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
            
            # Kiểm tra xem file đã có data chưa để in Header
            file_exists = os.path.isfile(output_txt_path) and os.path.getsize(output_txt_path) > 0
            
            # Lấy giá trị metric (nếu không có thì trả về 'N/A')
            m_acc = metric_one_dataset.get('acc', 'N/A')
            m_auc = metric_one_dataset.get('auc', 'N/A')
            m_eer = metric_one_dataset.get('eer', 'N/A')
            m_ap  = metric_one_dataset.get('ap', 'N/A')

            # Lấy weights path từ global args
            current_weight = args.weights_path if args.weights_path else "Unknown_Weight"
            
            with open(output_txt_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    # Ghi dòng tiêu đề, căn lề cực đẹp
                    f.write(f"{'Dataset':<60}\t{'ACC':<8}\t{'AUC':<8}\t{'EER':<8}\t{'AP':<8}\n")
                    f.write("-" * 105 + "\n")
                
                # Ghi data, format số thực lấy 4 chữ số thập phân cho gọn
                if isinstance(m_acc, float):
                    f.write(f"{key:<60}\t{m_acc:<8.4f}\t{m_auc:<8.4f}\t{m_eer:<8.4f}\t{m_ap:<8.4f}\t{current_weight:<70}\n")
                else:
                    f.write(f"{key:<60}\t{m_acc:<8}\t{m_auc:<8}\t{m_eer:<8}\t{m_ap:<8}\t{current_weight:<70}\n")
            # =================================================================

        threshold = 0.5
        pred_nps = [1 if i > threshold else 0 for i in predictions_nps]
        if write_confusion_matrix:
            cm = confusion_matrix(label_nps, pred_nps)
            print("Confusion Matrix:")
            print(cm)

        pred_0 = predictions_nps[label_nps == 0]
        pred_1 = predictions_nps[label_nps == 1]
        plt.figure()
        plt.ylim(0, len(predictions_nps) // 32)
        plt.hist(pred_0, bins=100, color='blue', alpha=0.5, label=f'Real {key}')
        plt.hist(pred_1, bins=100, color='yellow', alpha=0.5, label=f'Fake {key}')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Predictions for {key}")
        plt.legend()
        plt.savefig(f"./figures/hist_{key}_height{len(predictions_nps) // 32}.jpg", dpi=300, bbox_inches="tight")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


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
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    if config.get('confusion_matrix', False):
        global write_confusion_matrix
        write_confusion_matrix = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
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
        # print("ckpt keys:", ckpt.keys())
        model.load_state_dict(ckpt, strict=False) #check later, initially strict=True
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
