'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
from PIL import Image
import os
import numpy as np
import torch
import torchvision
from utils import *
from torch.utils.data import Dataset
from torchvision import transforms as T


__all__ = ('Dataset', 'Repeat')


# repeat a dataset until the length of dataset reaches "new_length"
class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


# construct dataset 
class Dataset(Dataset):
    def __init__(self, img_size, data_path, class_name, norm_mean, norm_std, is_pairset =False, is_train=True, init_chip = 0, fin_chip = -1):
        self.img_size = img_size
        self.dataset_path = data_path
        self.is_pairset = is_pairset
        self.class_name = class_name
        self.is_train = is_train
        self.init_chip = init_chip
        self.fin_chip = fin_chip

        self.x, self.y, self.mask, self.chip_list = self.load_dataset_folder()
        if init_chip ==0 and fin_chip ==-1:
            pass
        else:
            self.x, self.y, self.mask, self.chip_list = self.remove_files()

        self.transform_x = T.Compose([
            T.Resize(self.img_size, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor()])

        self.transform_mask = T.Compose([
            T.Resize(self.img_size, torchvision.transforms.InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(norm_mean, norm_std)])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x)
        x = np.array(x) 
        x = gray2rgb(x)
        #

        org_x = x

        # dataset for twin networks
        if self.is_pairset==True:
            if x.shape[1] > x.shape[0]:
                img_size = x.shape[0] 
                x1 = x[:, :img_size, :]
                x2 = x[:, img_size:img_size*2, :]
                mask1 = x[:, img_size*2:img_size*3, :]
                mask2 = x[:, img_size*3:img_size*4, :]
            elif x.shape[0] > x.shape[1]:
                img_size = x.shape[1] 
                x1 = x[:img_size, :, :]
                x2 = x[img_size:img_size*2, :, :]
                mask1 = x[img_size*2:img_size*3, :, :]
                mask2 = x[img_size*3:img_size*4, :, :]
            else:
                raise KeyboardInterrupt

            x1 = Image.fromarray(x1)
            x1 = self.transform_x(x1)
            x1 = self.normalize(x1)

            x2 = Image.fromarray(x2)
            x2 = self.transform_x(x2)
            x2 = self.normalize(x2)

            mask1 = Image.fromarray(mask1)
            mask1 = self.transform_mask(mask1)

            mask2 = Image.fromarray(mask2)
            mask2 = self.transform_mask(mask2)

            mask = torch.abs(mask1-mask2)
            return [x1,x2], y, mask 
        # dataset for single networks
        else:
            mask = self.mask[idx]
            x = Image.fromarray(x)
            x = self.transform_x(x)
            x = self.normalize(x)
            if y == 0:
                mask = torch.zeros([3, self.img_size[0], self.img_size[1]])
            else:
                if os.path.isfile(mask)==True:
                    mask = Image.open(mask)
                    mask = self.transform_mask(mask)
                else:
                    print(self.x[idx])
                    print(y)
                    raise KeyboardInterrupt

            return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask, chip_list= [], [], [], []

        phase = 'train' if self.is_train else 'test'
        if self.is_pairset==True:
            img_dir = os.path.join(self.dataset_path, 'pair_set', self.class_name, phase)
        else:
            img_dir = os.path.join(self.dataset_path, 'single_set', self.class_name, phase)
        if self.is_pairset==False:
            gt_dir = os.path.join(self.dataset_path, 'single_set', self.class_name, 'ground_truth')
        else:
            pass

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.JPEG')])

            chip_names = ['~'.join(f.split('~')[:-1])
                         for f in os.listdir(img_type_dir)
                         if f.endswith('.png') or f.endswith('.JPEG')]

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                if self.is_pairset==False:
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                     for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                else:
                    pass
            x.extend(img_fpath_list)
            chip_list.extend(chip_names)
            if len(x)!=len(y):
                print(len(x))
                print(len(y))
                print(img_type)
                raise KeyboardInterrupt
        assert len(x) == len(y), 'number of x and y should be same'
        chip_list = list(np.unique(chip_list)) 

        return list(x), list(y), list(mask), chip_list


    # remove file names which not in the test chips from total file lists
    def remove_files(self):
        x, y, mask, chip_list = [], [], [], self.chip_list[self.init_chip:self.fin_chip]

        for x_i in range(len(self.x)):
            x_ = self.x[x_i]
            in_ = False
            for chip_ in chip_list:
                if chip_ in x_:
                    in_=True
                else:
                    pass
            if in_==True:
                x.append(self.x[x_i])
                y.append(self.y[x_i])
                if self.is_pairset==False:
                    mask.append(self.mask[x_i])
                else:
                    pass
            else:
                pass

        assert len(x) == len(y), 'number of x and y should be same'
        return x, y, mask, chip_list
