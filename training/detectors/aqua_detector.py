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
import inspect

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='aqua')
class AquaDetector(AbstractDetector):
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

    def features(self, data_dict: dict) -> torch.Tensor:
        landmarks = data_dict.get('landmark', None)

        sig = inspect.signature(self.backbone.features)

        if 'landmarks' in sig.parameters and landmarks is not None:
            return self.backbone.features(data_dict['image'], landmarks)
        else:
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
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
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
