'''
# description: Class for the VLM_Transformers_Detector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation
'''

import os
import logging
import datetime
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='vlm')
class VLMDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        # print("self.backbone=", self.backbone)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        ## if donot load the pretrained weights, fail to get good results
        #state_dict = torch.load(config['pretrained'])
        #for name, weights in state_dict.items():
        #    if 'pointwise' in name:
        #        state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        #state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        #backbone.load_state_dict(state_dict, False)
        #logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self, config):
        # config example:
        # {
        #   "loss_func": "cross_entropy_weighted",
        #   "loss_kwargs": {"class_weights": [w0, w1], "label_smoothing": 0.0}
        # }

        loss_class = LOSSFUNC[config["loss_func"]]

        loss_kwargs = config.get("loss_kwargs", {})
        if loss_kwargs is None:
            loss_kwargs = {}

        loss_func = loss_class(**loss_kwargs)
        return loss_func

    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred     = self.classifier(features)

        features = features.to(torch.float32)
        pred     = pred.to(torch.float32)

        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        # print("datadict_name=", data_dict)
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

            # Save video names for computing video-level AUC
            # self.video_names.extend(data_dict['name'])
        return pred_dict

    def get_backbone(self):
        return self.backbone
