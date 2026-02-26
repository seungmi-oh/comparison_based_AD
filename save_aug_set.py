import os, random, copy, cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.io import write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from custom_datasets import * 
from utils import *
from args import get_args
from main import init_seeds

class_names = ['total'] 
c = get_args()
is_pairset = False
save_samples = False
save_aug_sets = True
c.img_size = (c.input_size, c.input_size)  # HxW format
c.crp_size = (c.input_size, c.input_size)  # HxW format
c.img_dims = [3] + list(c.img_size)
init_seeds(c.seed)

data_cfg = cfg.fromfile(os.path.join(c.data_cfg_dir, f'{c.data_settings}.py'))
c.train_data_path = os.path.join(c.data_path, data_cfg.product.train, data_cfg.layer_type.train, data_cfg.data_type.train)

if save_samples ==True:
    save_dir = c.train_data_path.replace('plain', 'aug_samples') 
else:
    if c.aug_ratio_train <1.0:
        save_dir = c.train_data_path.replace('plain', 'aug_data') 
    else:
        save_dir = c.train_data_path.replace('plain', 'aug_full_data')
    total_type_list = ['defect', 'good']
        

for cl in class_names:
    data_cfg.class_name.train = cl
    print(cl)

    if is_pairset ==True:
        save_dir = os.path.join(save_dir, 'pair_set', cl)
    else:
        save_dir = os.path.join(save_dir, 'single_set', cl)

    # save synthetic defect samples
    if save_samples ==True:
        train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, is_pairset, c.loss_type, True) 
        sample_num = min(50, len(train_dataset))
        print(f'save {sample_num} samples')
        save_aug_samples(train_dataset, sample_num, save_dir, c.input_size)
    else:
        pass

    # save synthetic defect dataset
    if save_aug_sets ==True:
        train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, is_pairset, c.loss_type) 
        total_num = len(train_dataset)
        shuffle_indices = list(range(total_num))
        j=0
        for repeat in range(c.repeat_num):
            random.shuffle(shuffle_indices)
            j+=1
            for defect_idx in range(len(total_type_list)):
                defect_name = total_type_list[defect_idx]
                if defect_name == 'good':
                    train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, 0, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, is_pairset, c.loss_type) 
                else:
                    train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, is_pairset, c.loss_type) 
                
                if c.is_train ==True:
                    good_num = int(len(train_dataset)*(1-c.aug_ratio_train))
                else: 
                    good_num = int(len(train_dataset)/2)
                init_idx = good_num
                fin_idx = int(len(train_dataset))
                if defect_idx ==len(total_type_list)-1: # good 
                    idx_list = shuffle_indices[:init_idx]
                else: # defect
                    idx_list = shuffle_indices[init_idx:fin_idx]
                std = torch.FloatTensor(c.norm_std)
                mean = torch.FloatTensor(c.norm_mean)
                std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
                mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
                if is_pairset ==True: 
                    print(f'{defect_name} pair dataset: {len(idx_list)} samples')
                    save_pair_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, c.is_train)
                else:
                    print(f'{defect_name} plain dataset: {len(idx_list)} samples')
                    save_single_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, c.is_train)



