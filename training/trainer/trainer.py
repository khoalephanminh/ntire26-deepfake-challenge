# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
from torch.utils.data.distributed import DistributedSampler
import shutil
FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.distributed as dist

import os
import cv2
import torch
import numpy as np

#-------------------------------------------------------------------------------------------------------------
# SAVE_DIR = "/raid/dtle/NTIRE26-DeepfakeDetection/misc/deroded_pmm02ab12_ddl_026/"
SAVE_DIR = "/raid/dtle/NTIRE26-DeepfakeDetection/misc/deroded_visualization/deroded_xxx/"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR) 
os.makedirs(SAVE_DIR, exist_ok=True)

# how many batches/images to dump (prevent huge IO)
MAX_SAVE_BATCHES = 20          # only save first N batches of the epoch
MAX_SAVE_PER_BATCH = 64         # save first K images in the batch

def denorm_to_uint8(x: torch.Tensor, mean, std):
    """
    x: Tensor (C,H,W) or (B,C,H,W). returns uint8 RGB numpy (H,W,3) or (B,H,W,3)
    """
    if x.is_cuda:
        x = x.detach().cpu()
    else:
        x = x.detach()

    mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=x.dtype).view(-1, 1, 1)

    if x.dim() == 4:
        # B,C,H,W
        out = []
        for i in range(x.shape[0]):
            yi = x[i] * std_t + mean_t
            yi = yi.clamp(0, 1)
            yi = (yi.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            out.append(yi)
        return np.stack(out, axis=0)
    elif x.dim() == 3:
        # C,H,W
        y = x * std_t + mean_t
        y = y.clamp(0, 1)
        y = (y.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return y
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")

def save_batch_images(tag, tensor_bchw, labels=None, epoch=0, iteration=0):
    """
    tensor_bchw: torch.Tensor [B,C,H,W]
    labels: torch.Tensor [B] optional
    """
    if not isinstance(tensor_bchw, torch.Tensor):
        return
    if tensor_bchw.dim() != 4 or tensor_bchw.shape[1] != 3:
        # skip non-RGB
        return

    b = min(tensor_bchw.shape[0], MAX_SAVE_PER_BATCH)
    mean = [0.481, 0.457, 0.408]
    std = [0.268, 0.261, 0.275]


    imgs = denorm_to_uint8(tensor_bchw[:b], mean=mean, std=std)

    for j in range(b):
        lab = int(labels[j].item()) if isinstance(labels, torch.Tensor) and labels is not None else -1
        fn = f"e{epoch:03d}_it{iteration:06d}_b{j:02d}_y{lab}_{tag}.png"
        path = os.path.join(SAVE_DIR, fn)

        # cv2 wants BGR
        bgr = cv2.cvtColor(imgs[j], cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

#-------------------------------------------------------------------------------------------------------------

def gather_array(array, dtype=torch.float32):
    """Gather numpy array from all processes."""
    tensor = torch.tensor(array, dtype=dtype, device='cuda')
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output.cpu().numpy()

def gather_string_list(str_list, max_len=300):
    """Gather list of strings across processes."""
    enc = [s.encode('utf-8') for s in str_list]
    arr = np.zeros((len(enc), max_len), dtype=np.uint8)
    for i, e in enumerate(enc):
        arr[i, :len(e)] = np.frombuffer(e, dtype=np.uint8)
    tensor = torch.tensor(arr, device='cuda')

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    gathered = torch.cat(gathered, dim=0).cpu().numpy()

    result = []
    for row in gathered:
        s = bytes(row[row > 0]).decode('utf-8')
        result.append(s)
    return result

def to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)

class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model_name = config['model_name']
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        self.finetune_path = config.get('finetune_path', '')
        self.is_finetune = os.path.exists(self.finetune_path)
        print("is_finetune=", self.is_finetune)
        if self.is_finetune:
            self.load_ckpt(self.finetune_path)
            self.logger.info(f"Finetune from {self.finetune_path}")
        
        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                saved = saved.state_dict()

            if type(self.model) is DDP:
                # add the prefix 'module.' to the keys if needed
                saved = {k.replace('backbone.', 'module.backbone.'): v for k, v in saved.items()}
                saved = {k.replace('module.module.', 'module.'): v for k, v in saved.items()}
                self.model.load_state_dict(saved)
            else:
                # remove the prefix 'module.' from the keys if needed
                saved = {k.replace('module.backbone.', 'backbone.'): v for k, v in saved.items()}
                #Missing key(s) in state_dict: "loss_func.class_weights". 
                #Unexpected key(s) in state_dict: "module.loss_func.class_weights". 
                saved = {k.replace('module.loss_func.class_weights', 'loss_func.class_weights'): v for k, v in saved.items()}
                self.model.load_state_dict(saved)
                

            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()


            return losses,predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))

        # ==========================================================
        # STRATEGY 1: DYNAMIC BACKBONE FREEZING
        # ==========================================================
        # Safely extract the model (handles PyTorch DDP wrapping automatically)
        if self.config.get('freeze_backbone_epochs', None) is not None:
            self.logger.info(f"Epoch {epoch}: Freezing backbone for the first {self.config['freeze_backbone_epochs']} epochs.")
            freeze_epochs = self.config['freeze_backbone_epochs']
            model_ref = self.model.module if type(self.model) is DDP else self.model
            
            if epoch < freeze_epochs:
                model_ref.get_backbone().set_backbone_trainable(False)
                self.logger.info(f"Epoch {epoch}: Backbone FROZEN. Forcing CNN to learn geometry.")
            elif epoch == freeze_epochs:
                model_ref.get_backbone().set_backbone_trainable(True)
                self.logger.info(f"Epoch {epoch}: Backbone UNFROZEN. Joint fine-tuning activated.")
        
        if self.config.get('freeze_backbone_iters', None):
            freeze_iters = self.config['freeze_backbone_iters']
        else:
            freeze_iters = 1_000_000_000
        # ==========================================================

        if self.config['ddp'] and isinstance(train_data_loader.sampler, DistributedSampler):
            print("Doing ddp..., setting epoch to ", epoch)
            train_data_loader.sampler.set_epoch(epoch)

        if epoch==0:
            times_per_epoch = 2
        # elif epoch==1: 
        elif epoch<=3: 
            times_per_epoch = 20 
        else:
            times_per_epoch = 20 
        times_per_epoch = 200

        if self.model_name == 'prodet' and epoch>=1:
            times_per_epoch = 3
        #times_per_epoch=4

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        # self.is_finetune = False
        total_iter = len(train_data_loader)
        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=total_iter, dynamic_ncols=True):
            #-------------------------------------------------------------------------------------------------------------
            # if epoch == 0 and iteration < MAX_SAVE_BATCHES:
            #     labels = data_dict.get("label", None)

            #     # Main image
            #     if "image" in data_dict:
            #         img = data_dict["image"]
            #         # Sometimes your pipeline may store a list; handle that too
            #         if isinstance(img, (list, tuple)):
            #             # e.g. [image, image_2, image_3]
            #             for k, t in enumerate(img):
            #                 if isinstance(t, torch.Tensor) and t.dim() == 4:
            #                     save_batch_images(f"image_list{k}", t, labels=labels, epoch=epoch, iteration=iteration)
            #         elif isinstance(img, torch.Tensor):
            #             save_batch_images("image", img, labels=labels, epoch=epoch, iteration=iteration)

            #     # Optional multi-res keys from collate_fn
            #     if "image_2" in data_dict and isinstance(data_dict["image_2"], torch.Tensor):
            #         save_batch_images("image_2", data_dict["image_2"], labels=labels, epoch=epoch, iteration=iteration)

            #     if "image_3" in data_dict and isinstance(data_dict["image_3"], torch.Tensor):
            #         save_batch_images("image_3", data_dict["image_3"], labels=labels, epoch=epoch, iteration=iteration)
            #-------------------------------------------------------------------------------------------------------------
            if iteration == freeze_iters:
                model_ref.get_backbone().set_backbone_trainable(True)
                self.logger.info(f"Epoch {epoch} Iteration {iteration}: Backbone UNFROZEN. Joint fine-tuning activated.")
                freeze_iters = 1_000_000_000


            # run test at the beginning if finetuning
            if self.is_finetune:
                self.is_finetune = False
                test_best_metric = None
                # if test_data_loaders is not None and (not self.config['ddp'] ):
                #     self.logger.info("===> Test start!")
                #     test_best_metric = self.test_epoch(
                #         epoch,
                #         iteration,
                #         test_data_loaders,
                #         step_cnt,
                #     )
                # elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                #     self.logger.info("===> Test start!")
                #     test_best_metric = self.test_epoch(
                #         epoch,
                #         iteration,
                #         test_data_loaders,
                #         step_cnt,
                #     )
                # else:
                #     test_best_metric = None


            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            with torch.no_grad():
                if type(self.model) is DDP:
                    batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
                else:
                    batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                # train_recorder_metric[name].update(value)
                if value is not None: # <-- Bỏ qua việc update nếu metric bị None
                    train_recorder_metric[name].update(to_float(value))  # convert to float for metric recorder, since some metric like AUC is not differentiable and thus is a python float instead of a tensor
            ## store loss
            for name, value in losses.items():
                # train_recorder_loss[name].update(value)
                train_recorder_loss[name].update(to_float(value))  # convert to float for loss recorder as well, for consistent display in tensorboard, since some loss may not be a single scalar tensor but a dict of multiple values

            # run tensorboard to visualize the training process
            if (iteration % 100 == 0 or iteration == total_iter-1) and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()

                # ==========================================================
                # NEW LOGGING: Gamma and Learning Rates
                # ==========================================================
                # 1. Safely extract Gamma (handles both DDP and normal models)
                gamma_val = None
                for name, param in self.model.named_parameters():
                    if 'landmark_gamma' in name:
                        gamma_val = param.item()
                        break
                
                if gamma_val is not None:
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), 'landmark_gamma')
                    writer.add_scalar('hyperparameters/landmark_gamma', gamma_val, global_step=step_cnt)
                    self.logger.info(f"Iter: {step_cnt}    training-hyperparam, landmark_gamma: {gamma_val:.6f}")

                # 2. Extract and log Learning Rates for all groups
                lr_str = f"Iter: {step_cnt}    "
                for i, param_group in enumerate(self.optimizer.param_groups):
                    lr_val = param_group['lr']
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), f'lr_group_{i}')
                    writer.add_scalar(f'hyperparameters/lr_group_{i}', lr_val, global_step=step_cnt)
                    lr_str += f"training-lr, group_{i}: {lr_val:.2e}    "
                self.logger.info(lr_str)
                # ==========================================================

                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

                self.save_ckpt('test', 'last_train', f"{epoch}+{iteration}") # Save latest train ckpt

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1

        if epoch==0:
            self.save_ckpt('test', 'train_ep0', f"{epoch}+{iteration}") # Save latest train ep0 ckpt
            
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader), dynamic_ncols=True):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    # test_recorder_loss[name].update(value)
                    test_recorder_loss[name].update(to_float(value))  # convert to float for loss recorder as well, for consistent display in tensorboard, since some loss may not be a single scalar tensor but a dict of multiple values

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        # print("len(data_loader):",len(test_data_loaders))
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')

            # # Gather across DDP processes: # Took forever to gather :<<<
            # if dist.is_initialized():
            #     predictions_nps = gather_array(predictions_nps)
            #     label_nps = gather_array(label_nps)
            #     # data_dict['image'] = gather_string_list(data_dict['image'])  # Optional if used in metric calc

            # #----------------------------------------------------------------------------------------------------------------------

            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)
            # print("epoch=",epoch,"key=",key)
            # if epoch < 5: break #delete later

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        
        # self.save_ckpt('test', 'last_test', f"{epoch}+{iteration}") # Save latest ckpt
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
