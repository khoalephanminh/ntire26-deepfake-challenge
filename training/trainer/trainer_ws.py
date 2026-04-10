# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
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
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import shutil
from metrics.utils import get_test_metrics
from contextlib import nullcontext

import cv2

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------------------------------------
# SAVE_DIR = "/raid/dtle/NTIRE26-DeepfakeDetection/misc/deroded_pmm02ab12_ddl_026/"
SAVE_DIR = "/raid/dtle/NTIRE26-DeepfakeDetection/misc/deroded_xxx/"
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

def to_float(v):
    if isinstance(v, torch.Tensor): return float(v.detach().cpu().item())
    return float(v)

class Trainer(object):
    def __init__(self, config, model, optimizer, scheduler, logger, metric_scoring='auc', time_now=None, swa_model=None):
        self.config = config
        self.model_name = config['model_name']
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {} 
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(lambda: defaultdict(lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf')))
        self.speed_up() 

        self.finetune_path = config.get('finetune_path', '')
        self.is_finetune = os.path.exists(self.finetune_path)
        if self.is_finetune:
            self.load_ckpt(self.finetune_path)
            self.logger.info(f"Finetune from {self.finetune_path}")
        
        self.timenow = time_now if time_now else datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
        self.log_dir = os.path.join(self.config['log_dir'], self.config['model_name'], self.config['model_name'] + task_str + '_' + self.timenow)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = os.path.join(self.log_dir, phase, dataset_key, metric_key, "metric_board")
            os.makedirs(writer_path, exist_ok=True)
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True, output_device=self.config['local_rank'])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            if type(self.model) is DDP:
                saved = {k.replace('backbone.', 'module.backbone.'): v for k, v in saved.items()}
                saved = {k.replace('module.module.', 'module.'): v for k, v in saved.items()}
                self.model.load_state_dict(saved)
            else:
                saved = {k.replace('module.backbone.', 'backbone.'): v for k, v in saved.items()}
                saved = {k.replace('module.loss_func.class_weights', 'loss_func.class_weights'): v for k, v in saved.items()}
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "ckpt_best.pth")
        if self.config['ddp'] == True: torch.save(self.model.state_dict(), save_path)
        else: torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file: pickle.dump(data_dict, file)

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file: pickle.dump(metric_one_dataset, file)

    def train_epoch(self, epoch, train_data_loader, test_data_loaders=None):
        self.logger.info("===> Epoch[{}] start!".format(epoch))

        # Handle Samplers setting epoch for DDP randomness
        # if self.config['ddp'] and hasattr(train_data_loader.sampler, 'set_epoch'):
        #     train_data_loader.sampler.set_epoch(epoch)
        if hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)

        accumulation_steps = self.config.get('accumulation_steps', 1)

        # test_step = len(train_data_loader) // (50 * accumulation_steps) #set to 50 for 7 datasets
        # Guarantee test_step is a perfect multiple of accumulation_steps to ensure alignment
        base_test_step = max(1, len(train_data_loader) // 100)
        test_step = max(accumulation_steps, (base_test_step // accumulation_steps) * accumulation_steps)
        
        step_cnt = epoch * len(train_data_loader)
        
        # --- STRATEGY: ACCUMULATION ---

        self.save_data_dict('train', train_data_loader.dataset.data_dict, ','.join(self.config['train_dataset']))
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        total_iter = len(train_data_loader)
        
        # Zero gradients at the very beginning
        self.optimizer.zero_grad()

        for iteration, data_dict in tqdm(enumerate(train_data_loader), total=total_iter, dynamic_ncols=True):
            # ================= QUICK DEBUG =================
            # if iteration <= 10 and self.config['local_rank'] == 0:
            #     self.logger.info("\n" + "="*40)
            #     self.logger.info("BATCH 0: RAW IMAGE PATHS")
            #     self.logger.info("="*40)
            #     # data_dict['path'] is a tuple of lists containing the strings
            #     for path_list in data_dict['path']:
            #         self.logger.info(path_list[0]) # print the exact file path
            #     self.logger.info("="*40 + "\n")
            # ===============================================

            # #-------------------------------------------------------------------------------------------------------------
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
            # #-------------------------------------------------------------------------------------------------------------

            self.setTrain()
            for key in data_dict.keys():
                # ONLY push actual PyTorch Tensors to the GPU
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()

            # Determine if we should synchronize gradients on this step
            is_sync_step = ((iteration + 1) % accumulation_steps == 0) or (iteration + 1 == total_iter)
            
            # Using no_sync to vastly speed up DDP during accumulation
            my_context = self.model.no_sync() if (type(self.model) is DDP and not is_sync_step) else nullcontext()

            with my_context:
                predictions = self.model(data_dict)
                if type(self.model) is DDP: losses = self.model.module.get_losses(data_dict, predictions)
                else: losses = self.model.get_losses(data_dict, predictions)
                
                # Normalize loss by accumulation steps
                loss = losses['overall'] / accumulation_steps
                loss.backward()

            if is_sync_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Step the scheduler PER ITERATION if requested (crucial for Cosine Warmup)
                if self.scheduler is not None and self.config.get('step_lr_per_iter', False):
                    self.scheduler.step()

            with torch.no_grad():
                if type(self.model) is DDP: batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
                else: batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                if value is not None: train_recorder_metric[name].update(to_float(value)) 
            for name, value in losses.items():
                train_recorder_loss[name].update(to_float(value)) 

            # Only log and test exactly on sync steps to avoid noisy sub-batch logs
            if is_sync_step and ((iteration + 1) % 100 == 0 or iteration == total_iter-1) and self.config['local_rank']==0:
                
                # Accurately report LR
                lr_str = f"Iter: {step_cnt}    "
                for i, param_group in enumerate(self.optimizer.param_groups):
                    lr_val = param_group['lr']
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), f'lr_group_{i}')
                    writer.add_scalar(f'hyperparameters/lr_group_{i}', lr_val, global_step=step_cnt)
                    lr_str += f"training-lr, group_{i}: {lr_val:.2e}    "
                self.logger.info(lr_str)

                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg != None:
                        loss_str += f"training-loss, {k}: {v_avg}    "
                        writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                        writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)

                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average() 
                    if v_avg != None:
                        metric_str += f"training-metric, {k}: {v_avg}    "
                        writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                        writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                for recorder in train_recorder_loss.values(): recorder.clear()
                for recorder in train_recorder_metric.values(): recorder.clear()
                self.save_ckpt('test', 'last_train', f"{epoch}+{iteration}")


            # if (step_cnt+1) % test_step == 0 and is_sync_step:
            # Use local 'iteration' instead of global 'step_cnt' so it fires perfectly every epoch
            if (iteration + 1) % test_step == 0 and is_sync_step:
                # Safely check rank. If DDP is off, we are implicitly rank 0.
                is_main_process = (dist.get_rank() == 0) if (self.config.get('ddp', False) and dist.is_initialized()) else True
                
                if test_data_loaders is not None and is_main_process:
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)
                else: test_best_metric = None

            step_cnt += 1

        if epoch==0:
            self.save_ckpt('test', 'train_ep0', f"{epoch}+{iteration}") # Save latest train ep0 ckpt

        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / max(len(judge[zero_num:]), 1)
        acc_real = np.count_nonzero(judge[:zero_num]) / max(len(judge[:zero_num]), 1)
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists, feature_lists, label_lists = [], [], []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), dynamic_ncols=True):
            if 'label_spe' in data_dict: data_dict.pop('label_spe') 
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0) 
            for key in data_dict.keys():
                # ONLY push actual PyTorch Tensors to the GPU
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
            
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())

            if type(self.model) is DDP: losses = self.model.module.get_losses(data_dict, predictions)
            else: losses = self.model.get_losses(data_dict, predictions)

            for name, value in losses.items():
                test_recorder_loss[name].update(to_float(value)) 

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('-inf') if self.metric_scoring != 'eer' else float('inf'))
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg': self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
            
        if losses_one_dataset_recorder is not None:
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg != None:
                    writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                    loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)

        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k in ['pred', 'label', 'dataset_dict']: continue
            metric_str += f"testing-metric, {k}: {v}    "
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        keys = test_data_loaders.keys()

        # Safely access the base model to check for WiSE-FT capabilities
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model = base_model.get_backbone()
        use_wise = getattr(base_model, "use_wise_ft", False)
        # print("trainer use_wise=", use_wise)
        
        # The Alpha Sweep Spectrum (1.0 = 100% Fine-Tuned, 0.1 = 90% Pre-Trained)
        alphas = [1.0]  # Default to just evaluating the fully fine-tuned model
        
        if iteration <= 1000 :  # In the early stages, we want to focus on the fully fine-tuned model
            alphas = [1.0] if use_wise else [1.0] 
        elif iteration <= 3000 :  # In the early stages, we want to focus on the fully fine-tuned model
            alphas = [1.0, 0.9] if use_wise else [1.0]
        # elif iteration <= 1500 :  # In the early stages, we want to focus on the fully fine-tuned model
        #     alphas = [1.0, 0.5, 0.3] if use_wise else [1.0]
        # elif iteration <= 5000:  # After 5000 iterations, we can afford a more fine-grained sweep
        #     alphas = [1.0, 0.9, 0.5, 0.3] if use_wise else [1.0]
        else: # After 10000 iterations, we can focus more on the pre-trained weights
            alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1] if use_wise else [1.0]
        
        if epoch > 0: 
            alphas = [1.0, 0.9, 0.5] if use_wise else [1.0]
            # alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1] if use_wise else [1.0]

        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            best_metric_val = float('-inf') if self.metric_scoring != 'eer' else float('inf')
            best_wise_metrics = None
            best_wise_losses = None
            best_alpha = 1.0

            if use_wise:
                self.logger.info("\n" + "-"*50)
                self.logger.info(f"--- STARTING WiSE-FT SWEEP FOR: {key} ---")

            # Sweep through the interpolation ratios
            for alpha in alphas:
                if use_wise:
                    if alpha == 1.0:
                        base_model.restore_ft_weights() # 100% FT (Standard)
                    else:
                        base_model.apply_wise_ft(alpha=alpha) # Interpolate in VRAM

                # Evaluate the current mathematical blend
                losses_one_dataset_recorder, predictions_nps, label_nps, _ = self.test_one_dataset(test_data_loaders[key])
                metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'])

                current_score = metric_one_dataset[self.metric_scoring]

                if use_wise:
                    self.logger.info(f"[*] Alpha: {alpha:.1f} -> {self.metric_scoring.upper()}: {current_score:.4f}")

                # Track the absolute best performing Alpha
                improved = (current_score > best_metric_val) if self.metric_scoring != 'eer' else (current_score < best_metric_val)
                if improved or best_wise_metrics is None:
                    best_metric_val = current_score
                    best_wise_metrics = metric_one_dataset
                    best_wise_losses = losses_one_dataset_recorder
                    best_alpha = alpha

            if use_wise:
                self.logger.info(f"--> WINNING ALPHA for {key}: {best_alpha:.1f} ({self.metric_scoring.upper()}: {best_metric_val:.4f})")
                self.logger.info("-" * 50 + "\n")

            # Accumulate the BEST alpha's metrics for the average calculation
            for metric_name, value in best_wise_metrics.items():
                if metric_name in avg_metric: avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = best_wise_metrics[self.metric_scoring]
            
            # Save to tensorboard and best checkpoint using the WINNING alpha
            self.save_best(epoch, iteration, step, best_wise_losses, key, best_wise_metrics)

        # VERY IMPORTANT: Clean up and restore 100% FT weights before the next training epoch starts
        if use_wise:
            base_model.restore_ft_weights()

        if len(keys) > 0 and self.config.get('save_avg', False):
            for key in avg_metric:
                if key != 'dataset_dict': avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)
            
        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time

    @torch.no_grad()
    def inference(self, data_dict):
        return self.model(data_dict, inference=True)