import os, random, copy, cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from utils import makedirs, t2np, gray2rgb
from torchvision.io import write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from custom_datasets import * 
from args import get_args

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# convert pytorch tensor or numpy array to cv2 format 
def t2cv2(x):
    if isinstance(x, np.ndarray)==True:
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
    else:
        x = x.type(torch.uint8)
        x = t2np(x)
        if (len(x.shape)==2) or (x.shape[0]==1):
            x = np.transpose([np.squeeze(x)]*3, (1,2,0))
        else:
            x = np.transpose(x, (1,2,0))
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
    return x


# save augmentation samples to check the synthetic defect generation process
def save_aug_samples(train_dataset, sample_num, save_dir, inp_size):
    margin = np.uint8(np.ones((inp_size,10,3))*255)

    for idx in range(sample_num):
        viz_list = train_dataset[idx]
        viz_list = list(viz_list)
        new_viz_list = []
        for j in range(len(viz_list)):
            viz_data = viz_list[j]
            if len(viz_data)==2:
               for viz_data_ in viz_data:
                    new_viz_list.append(t2cv2(viz_data_))
                    new_viz_list.append(margin)
            else:
                new_viz_list.append(t2cv2(viz_data))
                new_viz_list.append(margin)

        save_img = np.uint8(np.concatenate(new_viz_list[:-1],1))

        makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, f'sample{idx:02d}.png'), save_img)


# save single synthetic defect dataset to evaluate networks
def save_single_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, is_train=False):  
    file_list = train_dataset.image_paths
    for idx in idx_list:
        x,y, mask= train_dataset[idx]
#            mask[mask !=0] =255
        x = (x*std+mean)*255
        x = t2cv2(x)

        mask = t2cv2(mask*255)

        file_name = os.path.basename(file_list[idx])
        if is_train==True:
            train_type = 'train'
            gt_type = 'train_gt'
        else:
            train_type = 'test'
            gt_type = 'ground_truth'            
        makedirs(os.path.join(save_dir, train_type, defect_name))
        makedirs(os.path.join(save_dir, gt_type, defect_name))

        if np.sum(mask)>0:
            cv2.imwrite(os.path.join(save_dir, train_type, defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)
            cv2.imwrite(os.path.join(save_dir, gt_type, defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), mask)
        else:
            makedirs(os.path.join(save_dir, train_type, 'good'))
            cv2.imwrite(os.path.join(save_dir, train_type, 'good', f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)


# save pair synthetic defect dataset to evaluate networks
def save_pair_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, is_train=False):  
    file_list = train_dataset.image_paths
    for idx in idx_list:
        x,y, mask= train_dataset[idx]
        x1, x2 =x
        y1, y2 =mask

        x1 = (x1*std+mean)*255
        x1 = t2cv2(x1)

        x2 = (x2*std+mean)*255
        x2 = t2cv2(x2)

        y1 = t2cv2(y1*255)
        y2 = t2cv2(y2*255)
        mask = np.abs(y1-y2)

        save_img = np.uint8(np.concatenate([x1,x2,y1,y2], axis=1))

        if is_train==True:
            train_type = 'train'
        else:
            train_type = 'test'  
            
        file_name = os.path.basename(file_list[idx])
        makedirs(os.path.join(save_dir, train_type, defect_name))

        if np.sum(mask)>0:
            cv2.imwrite(os.path.join(save_dir, train_type, defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), save_img)
        else:
            makedirs(os.path.join(save_dir, train_type, 'good'))
            cv2.imwrite(os.path.join(save_dir, train_type, 'good', f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), save_img)
