# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# description: training code.

import os
import argparse
import cv2
import random
import datetime
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from collections import Counter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Sampler
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.LinearLR import LinearDecayLR
from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from dataset.noise_augment_wrapper import NoiseAugmentWrapper

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="")
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

def reserve_vram_gib(target_gib=40, safety_gib=2, device=0):
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    target_bytes = int(max(0, (target_gib - safety_gib)) * (1024**3))
    reserve = min(target_bytes, int(free * 0.95))  
    if reserve <= 0: return None
    guard = torch.empty(reserve, dtype=torch.uint8, device=f"cuda:{device}")
    print(f"Reserved ~{reserve/1024**3:.1f} GiB on cuda:{device}")
    import time; time.sleep(2)
    del guard

def init_seed(config):
    if config['manualSeed'] is None: config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

# ==============================================================================
# STRATEGY 1: DECOUPLED TEMPERATURE DDP SAMPLER
# Forces 50/50 Real/Fake split & scales massive datasets mathematically
# ==============================================================================
class DecoupledTemperatureDDPSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, seed=0, alpha=0.5):
        self.dataset = dataset
        self.batch_size = batch_size # Local batch size per GPU
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.epoch = 0
        self.seed = seed
        self.alpha = alpha

        labels = np.array(dataset.data_dict['label'])
        datasets = np.array(dataset.data_dict['dataset_name'])

        self.real_indices = np.where(labels == 0)[0]
        self.fake_indices = np.where(labels == 1)[0]

        self.real_weights = self._compute_weights(self.real_indices, datasets)
        self.fake_weights = self._compute_weights(self.fake_indices, datasets)

        self.num_samples_per_replica = len(dataset) // self.num_replicas
        self.num_batches = self.num_samples_per_replica // self.batch_size
        
        # Force exact 50/50 balance in every single batch
        self.reals_per_batch = self.batch_size // 2
        self.fakes_per_batch = self.batch_size - self.reals_per_batch

    def _compute_weights(self, indices, datasets):
        subset_datasets = datasets[indices]
        counts = Counter(subset_datasets)
        weights = np.zeros(len(indices), dtype=np.float64)
        for i, ds in enumerate(subset_datasets):
            # Square root suppression for massive datasets
            weights[i] = 1.0 / (counts[ds] ** self.alpha)
        return torch.tensor(weights, dtype=torch.float64)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        total_reals_needed = self.num_batches * self.num_replicas * self.reals_per_batch
        total_fakes_needed = self.num_batches * self.num_replicas * self.fakes_per_batch

        sampled_reals = torch.multinomial(self.real_weights, total_reals_needed, replacement=True, generator=g)
        sampled_fakes = torch.multinomial(self.fake_weights, total_fakes_needed, replacement=True, generator=g)

        sampled_reals = torch.tensor(self.real_indices)[sampled_reals]
        sampled_fakes = torch.tensor(self.fake_indices)[sampled_fakes]

        my_reals = sampled_reals[self.rank :: self.num_replicas]
        my_fakes = sampled_fakes[self.rank :: self.num_replicas]

        batches = []
        for i in range(self.num_batches):
            b_reals = my_reals[i*self.reals_per_batch : (i+1)*self.reals_per_batch]
            b_fakes = my_fakes[i*self.fakes_per_batch : (i+1)*self.fakes_per_batch]
            batch = torch.cat([b_reals, b_fakes])
            batch = batch[torch.randperm(self.batch_size, generator=g)]
            batches.extend(batch.tolist())

        return iter(batches)

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch



class DecoupledTemperatureSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0, alpha=0.5):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.epoch = 0
        self.seed = seed
        self.alpha = alpha

        labels = np.array(dataset.data_dict['label'])
        datasets = np.array(dataset.data_dict['dataset_name'])

        self.real_indices = np.where(labels == 0)[0]
        self.fake_indices = np.where(labels == 1)[0]

        self.real_weights = self._compute_weights(self.real_indices, datasets)
        self.fake_weights = self._compute_weights(self.fake_indices, datasets)

        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // self.batch_size
        
        # Force exact 50/50 balance in every single batch
        self.reals_per_batch = self.batch_size // 2
        self.fakes_per_batch = self.batch_size - self.reals_per_batch

    def _compute_weights(self, indices, datasets):
        subset_datasets = datasets[indices]
        counts = Counter(subset_datasets)
        weights = np.zeros(len(indices), dtype=np.float64)
        for i, ds in enumerate(subset_datasets):
            # Square root suppression for massive datasets
            weights[i] = 1.0 / (counts[ds] ** self.alpha)
        return torch.tensor(weights, dtype=torch.float64)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        total_reals_needed = self.num_batches * self.reals_per_batch
        total_fakes_needed = self.num_batches * self.fakes_per_batch

        sampled_reals = torch.multinomial(self.real_weights, total_reals_needed, replacement=True, generator=g)
        sampled_fakes = torch.multinomial(self.fake_weights, total_fakes_needed, replacement=True, generator=g)

        my_reals = torch.tensor(self.real_indices)[sampled_reals]
        my_fakes = torch.tensor(self.fake_indices)[sampled_fakes]

        batches = []
        for i in range(self.num_batches):
            b_reals = my_reals[i*self.reals_per_batch : (i+1)*self.reals_per_batch]
            b_fakes = my_fakes[i*self.fakes_per_batch : (i+1)*self.fakes_per_batch]
            batch = torch.cat([b_reals, b_fakes])
            
            # Shuffle the 50/50 mix inside the batch so the network doesn't memorize patterns
            batch = batch[torch.randperm(self.batch_size, generator=g)]
            batches.extend(batch.tolist())

        return iter(batches)

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

# ==============================================================================


def prepare_training_data(config):
    # [Omitted standard baseline instantiations for brevity]
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')

    if config.get("use_degradation", False):
        cfg_detector = config.get("degradation", {})
        degradation_config = {
            **cfg_detector,
            "type": cfg_detector.get("type", "ours"),
            "global_p": cfg_detector.get("global_p", 0.8),
            "val_global_p": cfg_detector.get("val_global_p", 0.0),
            "op_p": cfg_detector.get("op_p", 0.5),
            "degradations_strength": cfg_detector.get("degradations_strength", 0.8),
            "use_beta": cfg_detector.get("use_beta", True),
            "distractor_p": cfg_detector.get("distractor_p", 0.15),
            "a": cfg_detector.get("a", 1.2),
            "b": cfg_detector.get("b", 1.2),
            "mean": config.get("mean", [0.485, 0.456, 0.406]), 
            "std":  config.get("std",  [0.229, 0.224, 0.225]),
        }
        train_set = NoiseAugmentWrapper(train_set, degradation_config, split="train")

    # --- INJECT CUSTOM DECOUPLED SAMPLER ---
    if config.get('use_decoupled_sampler', False):
        if config.get('ddp', False):
            print("=> Activating Decoupled Temperature DDP Sampler (50/50 Rigid Balance - Multi-GPU)")
            sampler = DecoupledTemperatureDDPSampler(
                dataset=train_set.dataset, 
                batch_size=config['train_batchSize'],
                seed=config.get('manualSeed', 1024),
                alpha=0.5
            )
        else:
            print("=> Activating Decoupled Temperature Sampler (50/50 Rigid Balance - Single-GPU)")
            sampler = DecoupledTemperatureSampler(
                dataset=train_set.dataset, 
                batch_size=config['train_batchSize'],
                seed=config.get('manualSeed', 1024),
                alpha=0.5 
            )

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
            drop_last=True
        )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=config['train_batchSize'],
            num_workers=int(config['workers']), collate_fn=train_set.collate_fn, sampler=sampler
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=config['train_batchSize'], shuffle=True,
            num_workers=int(config['workers']), collate_fn=train_set.collate_fn
        )
    return train_data_loader

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()  
        config['test_dataset'] = test_name  
        test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=config['test_batchSize'], shuffle=False,
            num_workers=int(config['workers']), collate_fn=test_set.collate_fn,
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    # [Other optimizers omitted for brevity]
    return optimizer


def choose_scheduler(config, optimizer, total_steps=None):
    if config['lr_scheduler'] is None: return None
    elif config['lr_scheduler'] == 'linear':
        return LinearDecayLR(optimizer, config['nEpochs'], int(config['nEpochs']*0.75))
    
    # --- STRATEGY: COSINE WARMUP ---
    elif config['lr_scheduler'] == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_iters = config.get('warmup_steps', 2000)
        
        # Protect against 1e-4 shock on the giant parameters
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_iters), eta_min=1e-6)
        
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
        config['step_lr_per_iter'] = True # Guarantee per-iteration steps
        return scheduler
    else:
        raise NotImplementedError('Scheduler not implemented')

def choose_metric(config): return config['metric_scoring']

def main():
    reserve_vram_gib(target_gib=80, safety_gib=2, device=args.local_rank)
    with open(args.detector_path, 'r') as f: config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f: config2 = yaml.safe_load(f)
    if 'label_dict' in config: config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    config['dataset_json_folder'] = 'preprocessing/dataset_json'
    
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path = os.path.join(config['log_dir'], config['model_name'], config['model_name'] + task_str + '_' + timenow)
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)
    
    init_seed(config)
    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    optimizer = choose_optimizer(model, config)

    # Calculate exact total steps for Cosine Scheduler
    accumulation_steps = config.get('accumulation_steps', 1)
    total_steps = (len(train_data_loader) // accumulation_steps) * config['nEpochs']
    scheduler = choose_scheduler(config, optimizer, total_steps=total_steps)

    metric_scoring = choose_metric(config)
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(epoch=epoch, train_data_loader=train_data_loader, test_data_loaders=test_data_loaders)
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    for writer in trainer.writers.values(): writer.close()

if __name__ == '__main__':
    main()