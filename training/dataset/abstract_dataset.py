# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# description: Abstract Base Class for all types of deepfake datasets.

from pathlib import Path
import sys
import lmdb
import os
import yaml
import json
import pickle 
import numpy as np
from collections import Counter
from copy import deepcopy
import cv2
import random
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
import albumentations as A

from .albu import IsotropicResize
from .fsbi_utils import get_dwt

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True

class DeepfakeAbstractBaseDataset(data.Dataset):
    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        if self.config.get('with_landmark', False) and self.config.get('landmark_dict_path', None) is not None:
            dict_path = self.config['landmark_dict_path']
            try:
                print(f"Loading binary landmarks from {dict_path}...")
                with open(dict_path, 'rb') as f:
                    self.landmark_dict = pickle.load(f)
                print("Landmarks loaded instantly!")
            except Exception as e:
                raise ValueError(f"Landmark dictionary file {dict_path} not found or invalid.")

        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        
        self.image_list = []
        self.label_list = []
        self.dataset_name_list = [] 
        
        if mode == 'train':
            dataset_list = config['train_dataset']
            image_list, label_list, video_name_list, dataset_name_list = [], [], [], []
            for one_data in dataset_list:
                # FIX: Capture tmp_vname instead of discarding it with '_'
                tmp_image, tmp_label, tmp_vname, tmp_ds_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                video_name_list.extend(tmp_vname)
                dataset_name_list.extend(tmp_ds_name) 
        elif mode == 'test':
            # FIX: Iterate test dataset just in case it is a list
            dataset_list = config['test_dataset']
            if not isinstance(dataset_list, list): dataset_list = [dataset_list]
            image_list, label_list, video_name_list, dataset_name_list = [], [], [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_vname, tmp_ds_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                video_name_list.extend(tmp_vname)
                dataset_name_list.extend(tmp_ds_name) 
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
            'dataset_name': dataset_name_list 
        }

        # ====================================================
        # NEW: DETAILED DATASET DISTRIBUTION PRINTER
        # ====================================================
        print("\n" + "="*52)
        print(f"      NTIRE 2026 '{self.mode.upper()}' DATASET DISTRIBUTION      ")
        print("      (Constraint: Dynamic Frame Strategy)      ")
        print("="*52)

        global_real = sum(1 for l in self.label_list if int(l) == 0)
        global_fake = sum(1 for l in self.label_list if int(l) == 1)
        global_total = len(self.label_list)
        global_unique = len(set(video_name_list))

        print(f"\nGlobal {self.mode.capitalize()} Effective Frames: {global_total:,}")
        print(f"  -> Total Real Frames: {global_real:,}")
        print(f"  -> Total Fake Frames: {global_fake:,}")
        print(f"Global {self.mode.capitalize()} Videos/Images (Unique IDs): {global_unique:,}\n")

        # Calculate Local Stats
        ds_stats = {}
        for ds, lbl, vid in zip(dataset_name_list, self.label_list, video_name_list):
            if ds not in ds_stats:
                ds_stats[ds] = {'real': 0, 'fake': 0, 'vids': set()}
            if int(lbl) == 0:
                ds_stats[ds]['real'] += 1
            else:
                ds_stats[ds]['fake'] += 1
            ds_stats[ds]['vids'].add(vid)

        # Print Local Stats
        for ds, stats in ds_stats.items():
            ds_total = stats['real'] + stats['fake']
            ds_unique = len(stats['vids'])
            print(f"--- {ds.upper()} ---")
            print(f"Total Effective {self.mode.capitalize()} Frames : {ds_total:,}")
            print(f"Total Unique Videos/Images   : {ds_unique:,}")
            print(f"  - Real Frames: {stats['real']:,}")
            print(f"  - Fake Frames: {stats['fake']:,}\n")
        print("="*52 + "\n")
        # ====================================================
        
        labels = np.asarray([int(x) for x in self.label_list]).astype(int)
        cnt = Counter(labels.tolist())
        n_real = cnt.get(0, 0)
        n_fake = cnt.get(1, 0)
        total = n_real + n_fake

        print(f"Total={total:,} | Real(0)={n_real:,} ({n_real/total:.2%}) | Fake(1)={n_fake:,} ({n_fake/total:.2%})")
        if self.mode == 'train':
            self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR)
            ], p = 0 if self.config['with_landmark']  else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=self.config['data_aug'].get('color_prob', 0.5)),
            A.ImageCompression(
                quality_lower=self.config['data_aug']['quality_lower'],
                quality_upper=self.config['data_aug']['quality_upper'], 
                p=self.config['data_aug'].get('jpeg_prob', 0.5)
            )
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) if self.config['with_landmark'] else None)
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        return landmarks * scale_factor

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        label_list = []
        frame_path_list = []
        video_name_list = []
        dataset_name_list = [] 

        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace('/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public')
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            raise ValueError(f'dataset {dataset_name} not exist!')

        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'

        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']

            cnt_skip = 0
            cnt_skip_landmark = 0
            cnt_label_real = 0
            cnt_label_fake = 0
            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = video_info['label'] + '_' + video_name
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found.')
                label_idx = self.config['label_dict'][video_info['label']]
                
                # ==========================================================
                # NEW: STOCHASTIC SUBSAMPLING GATEKEEPER (COLLECTION PHASE)
                # ==========================================================
                if self.mode == 'train':
                    drop_rates = self.config.get('stochastic_drop_rate', {})
                    if dataset_name in drop_rates:
                        drop_config = drop_rates[dataset_name]
                        drop_prob = 0.0
                        
                        # Handle Class-Specific Drops (e.g., RedFace: {1: 0.8})
                        if isinstance(drop_config, dict):
                            if label_idx in drop_config:
                                drop_prob = float(drop_config[label_idx])
                            elif str(label_idx) in drop_config: # Catch yaml string keys
                                drop_prob = float(drop_config[str(label_idx)])
                        # Handle Global Dataset Drops (e.g., DDL: 0.75)
                        else:
                            drop_prob = float(drop_config)

                        # Execute the probability drop at the video level
                        if drop_prob > 0.0 and random.random() < drop_prob:
                            continue # Skip this video and its frames entirely!
                # ==========================================================

                if label_idx == 0: cnt_label_real += 1
                elif label_idx == 1: cnt_label_fake += 1
                frame_paths = video_info['frames']

                if self.config.get('with_landmark', False):
                    missing_landmark = False
                    for f_path in frame_paths:
                        full_path = f_path if f_path.startswith('.') else os.path.join(f'./{self.config["rgb_dir"]}', f_path).replace('\\', '/')
                        if not self.check_landmark_existence(full_path):
                            missing_landmark = True
                            break
                    if missing_landmark:
                        cnt_skip_landmark += 1
                        continue

                if len(frame_paths) == 0:
                    cnt_skip += 1
                    continue
                if '\\' in frame_paths[0]:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                else:
                    frame_paths = sorted(
                        frame_paths,
                        key=lambda x: int(''.join(reversed(''.join(
                            __import__("itertools").takewhile(str.isdigit, reversed(x.split('/')[-1].split('.')[0]))
                        ))) or "0")
                    )

                # total_frames = len(frame_paths)
                # if self.frame_num < total_frames:
                #     total_frames = self.frame_num
                #     if self.video_level:
                #         start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                #         frame_paths = frame_paths[start_frame:start_frame + self.frame_num] 
                #     else:
                #         step = total_frames // self.frame_num
                #         frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]

                # --- NEW DYNAMIC FRAME TRICK (Mimicking Original Logic) ---
                current_frame_num = self.frame_num
                
                # Check if yaml config defines a specific cap for this dataset & label
                dynamic_caps = self.config.get('dynamic_frame_cap', {})
                if dataset_name in dynamic_caps:
                    if label_idx in dynamic_caps[dataset_name]:
                        current_frame_num = dynamic_caps[dataset_name][label_idx]
                    elif str(label_idx) in dynamic_caps[dataset_name]: # Catch yaml dict string keys
                        current_frame_num = dynamic_caps[dataset_name][str(label_idx)]

                total_frames = len(frame_paths)
                if current_frame_num < total_frames:
                    total_frames = current_frame_num  # Overwrites total_frames (Original Behavior)
                    step = total_frames // current_frame_num  # Evaluates to 1
                    frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:current_frame_num]
                # ----------------------------------------------------------

                label_list.extend([label_idx] * total_frames)
                frame_path_list.extend(frame_paths)
                video_name_list.extend([unique_video_name] * len(frame_paths))
                dataset_name_list.extend([dataset_name] * len(frame_paths)) 

        shuffled = list(zip(label_list, frame_path_list, video_name_list, dataset_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list, dataset_name_list = zip(*shuffled)

        return frame_path_list, label_list, video_name_list, dataset_name_list

    def load_rgb(self, file_path):
        size = self.config['resolution']
        center_crop = self.config.get('center_crop_size', None) # <--- Read the new config

        if not file_path[0] == '.':
            file_path = os.path.join(f'./{self.config["rgb_dir"]}', file_path).replace('\\', '/')
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- NEW: ZERO-INTERPOLATION CENTER CROP ---
        if center_crop is not None:
            h, w = img.shape[:2]
            # Calculate exact center slicing coordinates
            start_y = max(0, (h - center_crop) // 2)
            start_x = max(0, (w - center_crop) // 2)
            
            # Pure array slicing (NO BLUR!)
            img = img[start_y:start_y + center_crop, start_x:start_x + center_crop]
            
            # Safety fallback: if the original image was somehow smaller than 252
            if img.shape[0] != center_crop or img.shape[1] != center_crop:
                raise ValueError(f"Center crop failed for {file_path}. Expected size: {center_crop}, Got: {img.shape[:2]}")
                img = cv2.resize(img, (center_crop, center_crop), interpolation=cv2.INTER_CUBIC)

        else: #- ORIGINAL: FALLBACK RESIZING LOGIC (KEPT IN CASE CENTER CROP IS NOT USED)     
            if img.shape[0] > size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            elif img.shape[0] < size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                
        return Image.fromarray(np.array(img, dtype=np.uint8))


    def load_mask(self, file_path):
        size = self.config['resolution']
        if file_path is None: return np.zeros((size, size, 1))
        if not file_path[0] == '.':
            file_path = os.path.join(f'./{self.config["rgb_dir"]}', file_path).replace('\\', '/')
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None: mask = np.zeros((size, size))
        else:
            return np.zeros((size, size, 1))
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark_by_dict(self, image_path):
        if image_path in self.landmark_dict:
            landmark = np.array(self.landmark_dict[image_path], dtype=np.float32)
            landmark = self.rescale_landmarks(landmark, original_size=256, new_size=self.config['resolution'])
            return landmark
        return None

    def check_landmark_in_dict(self, image_path):
        return image_path in self.landmark_dict

    def load_landmark(self, file_path):
        if file_path is None: return np.zeros((81, 2))
        if not file_path[0] == '.':
            file_path = os.path.join(f'./{self.config["rgb_dir"]}', file_path).replace('\\', '/')
        if os.path.exists(file_path):
            landmark = np.load(file_path, allow_pickle=True)
            assert landmark.shape == (81, 2), f"Landmark shape is not (81, 2) for file {file_path}"
            landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        else:
            return np.zeros((81, 2))
        return landmark

    def load_landmark_agnostic(self, image_path):
        assert self.config.get('with_landmark', False)
        full_path = image_path if image_path.startswith('.') else os.path.join(f'./{self.config["rgb_dir"]}', image_path).replace('\\', '/')
        if not '/raid/dtle/NTIRE26-DeepfakeDetection/' in full_path:
            full_path = full_path.replace('./datasets/', '/raid/dtle/NTIRE26-DeepfakeDetection/datasets/')
        if self.config.get('landmark_dict_path', None) is not None:
            return self.load_landmark_by_dict(full_path)
        landmark_path = self.get_landmark_path(full_path)
        if not os.path.exists(landmark_path):
            raise ValueError(f"Landmark file {landmark_path} does not exist.")
        return self.load_landmark(landmark_path)

    def check_landmark_existence(self, image_path):
        full_path = image_path if image_path.startswith('.') else os.path.join(f'./{self.config["rgb_dir"]}', image_path).replace('\\', '/')
        if not '/raid/dtle/NTIRE26-DeepfakeDetection/' in full_path:
            full_path = full_path.replace('./datasets/', '/raid/dtle/NTIRE26-DeepfakeDetection/datasets/')
        assert self.config.get('with_landmark', False)
        if self.config.get('landmark_dict_path', None) is not None:
            return self.check_landmark_in_dict(full_path)
        landmark_path = self.get_landmark_path(full_path)
        return os.path.exists(landmark_path)

    def to_tensor(self, img): return T.ToTensor()(img)
    def normalize(self, img): return T.Normalize(mean=self.config['mean'], std=self.config['std'])(img)
    def normalize_custom(self, img, mean, std): return T.Normalize(mean=mean, std=std)(img)

    def get_landmark_path(self, image_path):
        if 'frame' in image_path:
            ret = image_path.replace('frames', 'landmarks') 
        else:
            foldername = os.path.dirname(image_path).split('/')[-1]
            ret = image_path.replace(foldername, f'{foldername}_landmarks')
        return os.path.join(os.path.dirname(ret), Path(ret).stem + '.npy')

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        kwargs = {'image': img}
        if landmark is not None:
            kwargs['keypoints'] = [tuple(p) for p in landmark]
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0: kwargs['mask'] = mask

        transformed = self.transform(**kwargs)
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        if augmented_landmark is not None: augmented_landmark = np.array(augmented_landmark)
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()
        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        if not isinstance(image_paths, list): image_paths = [image_paths]

        image_tensors, landmark_tensors, mask_tensors = [], [], []
        augmentation_seed = None

        for image_path in image_paths:
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)
            mask_path = image_path.replace('frames', 'masks')
            landmark_path = self.get_landmark_path(image_path)  
            
            if image_path.endswith('.gif'): return self.__getitem__(0)
            try: image = self.load_rgb(image_path)
            except Exception as e: return self.__getitem__(0)
            
            image = np.array(image)
            mask = self.load_mask(mask_path) if self.config['with_mask'] else None
            landmarks = self.load_landmark(landmark_path) if self.config['with_landmark'] else None

            if self.mode == 'train' and self.config['use_data_augmentation']:
                try: image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
                except Exception as e: return self.__getitem__(random.randint(0, len(self.data_dict['image']) - 1))
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
            
            if self.config['model_name'] == 'fsbi':
                image_size = (self.config['resolution'], self.config['resolution'])
                image_trans = get_dwt(image_trans, image_size)

            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']: landmarks_trans = torch.from_numpy(landmarks_trans.astype(np.float32))
                if self.config['with_mask']: mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        # --- THE FIX: ALWAYS EXTRACT CLEAN NONE OBJECTS, NO LISTS CONTAINING NONE ---
        if self.video_level:
            image_tensors = torch.stack(image_tensors, dim=0)
            if not any(l is None for l in landmark_tensors): 
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            else: 
                landmark_tensors = None
                
            if not any(m is None for m in mask_tensors): 
                mask_tensors = torch.stack(mask_tensors, dim=0)
            else: 
                mask_tensors = None
        else:
            image_tensors = image_tensors[0]
            landmark_tensors = landmark_tensors[0]
            mask_tensors = mask_tensors[0]

        # return image_tensors, label, landmark_tensors, mask_tensors
        # ADD image_paths TO THE RETURN
        return image_tensors, label, landmark_tensors, mask_tensors, image_paths
    
    @staticmethod
    def collate_fn(batch):
        # images, labels, landmarks, masks = zip(*batch)
        # UNPACK paths HERE
        images, labels, landmarks, masks, paths = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # This will now successfully evaluate to True and cast to None
        if not any(l is None for l in landmarks): landmarks = torch.stack(landmarks, dim=0)
        else: landmarks = None

        if not any(m is None for m in masks): masks = torch.stack(masks, dim=0)
        else: masks = None

        # return {'image': images, 'label': labels, 'landmark': landmarks, 'mask': masks}
        # ADD 'path': paths to the return dict
        return {'image': images, 'label': labels, 'landmark': landmarks, 'mask': masks, 'path': paths}

    def __len__(self):
        return len(self.image_list)