'''This code is based on the DRAEM project (source: https://github.com/VitjanZ/DRAEM).
We modified and added the necessary modules or functions for our purposes.'''
import os, copy
import numpy as np
import torch
import cv2, math
import glob
import imgaug.augmenters as iaa
from utils import *
from torch.utils.data import Dataset
from torchvision import transforms as T
from .perlin import rand_perlin_2d_np

class PerlinDefectsTrainDataset(Dataset):
    def __init__(self, root_dir, class_name, norm_mean, norm_std, aug_ratio, use_in_domain_data, resize_shape, anomaly_size, anomaly_confidence, is_pairset=False, loss_type='cls', for_check =False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.is_pairset=is_pairset
        self.anomaly_size = anomaly_size
        self.anomaly_confidence = anomaly_confidence

        if self.is_pairset==True:
            img_dir = os.path.join(root_dir, 'pair_set', class_name, 'train')
        else:
            img_dir = os.path.join(root_dir, 'single_set', class_name, 'train')
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, 'good', '*.png')))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.normalize = T.Compose([T.Normalize(norm_mean, norm_std)])
        self.aug_ratio =aug_ratio
        self.use_in_domain_data = use_in_domain_data

        self.perlin_scale = 5
        self.min_perlin_scale = 3

        self.loss_type = loss_type
        self.for_check = for_check

    def __len__(self):
        return len(self.image_paths)

    # randomly transform the anomaly source image.
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]]])
        return aug

    # generate synthetic defect data 
    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        org_image = copy.deepcopy(image)

        # get an irregular pattern mask 
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = (perlin_noise-np.min(perlin_noise))/(np.max(perlin_noise)-np.min(perlin_noise)+1e-8)
        perlin_noise = self.rot(image=perlin_noise)
        anomalies_for_perlin_noise = np.expand_dims(1-perlin_noise, axis=-1)

        threshold = self.anomaly_size
        perlin_thr = copy.deepcopy(perlin_noise)
        perlin_thr = np.where(perlin_thr > threshold, np.ones_like(perlin_thr), np.zeros_like(perlin_thr))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        perlin_mask = copy.deepcopy(perlin_thr)

        # make the input mask 
        input_mask = copy.deepcopy(perlin_thr)
        kernel = np.ones((3, 3), np.uint8)
        input_mask = cv2.dilate(255-np.uint8(input_mask*255), kernel, 2)
        input_mask_org = cv2.GaussianBlur(input_mask, (5, 5), sigmaX = 0.3)
        defect_seg_nums, defect_seg_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(input_mask_org<255)) 
        sigma = 0.7+torch.rand(1).numpy()[0]*0.3
        input_mask = cv2.GaussianBlur(input_mask, (5, 5), sigmaX = sigma)
        input_mask = np.float32(input_mask)
        input_mask = np.expand_dims(input_mask/255.0, axis=-1)


        # make synthetic defects by blending with other imamges
        if len(anomaly_source_path)>0:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            if self.is_pairset==True:
                if anomaly_source_img.shape[1] > anomaly_source_img.shape[0]:
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1]*4, self.resize_shape[0]))
                    select = torch.rand(1).numpy()[0]
                    if select<0.5:
                        anomaly_source_img = anomaly_source_img[:, :self.resize_shape[1], :]
                    else:
                        anomaly_source_img = anomaly_source_img[:, self.resize_shape[1]:self.resize_shape[1]*2, :]
                elif anomaly_source_img.shape[0] > anomaly_source_img.shape[1]:
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]*4))
                    select = torch.rand(1).numpy()[0]
                    if select<0.5:
                        anomaly_source_img = anomaly_source_img[:self.resize_shape[0], :, :]
                    else:
                        anomaly_source_img = anomaly_source_img[self.resize_shape[0]:self.resize_shape[0]*2, :, :]
                else:
                    raise KeyboardInterrupt
            else:
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

            anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
            anomaly_source_img_augmented = aug(image=anomaly_source_img)
        # make synthetic defects by inserting random noise 
        else:
            beta = torch.rand(1).numpy()[0]*0.8
            anomaly_source_img = np.uint8(anomalies_for_perlin_noise*perlin_thr* beta*255)
            anomaly_source_img_augmented = copy.deepcopy(anomaly_source_img)
            anomaly_source_img_augmented[anomaly_source_img==0]=0.5

        anomaly_source_img_augmented= cv2.GaussianBlur(anomaly_source_img_augmented, (5, 5), sigmaX = 1)
        if len(anomaly_source_img_augmented.shape)==2:
            anomaly_source_img_augmented = np.expand_dims(anomaly_source_img_augmented, axis=-1)
        else:
            pass
        anomaly_source_img_augmented = anomaly_source_img_augmented.astype(np.float32)/255.0
        anomaly_mask = copy.deepcopy(np.uint8(perlin_thr*255))
        anomaly_mask = cv2.GaussianBlur(anomaly_mask, (5,5), sigmaX=1)
        anomaly_mask = np.expand_dims(anomaly_mask, axis=-1)
        anomaly_mask = anomaly_mask.astype(np.float32)/255.0
        anomaly_samples = anomaly_source_img_augmented.astype(np.float32) * anomaly_mask

        intensity_anomaly_diff = gray2rgb(np.abs(rgb2gray(org_image.astype(np.float32)*anomaly_mask) - rgb2gray(gray2rgb(anomaly_samples)))) 
        color_anomaly_diff = np.mean(np.abs(org_image.astype(np.float32)*anomaly_mask - gray2rgb(anomaly_samples)), axis=-1, keepdims=True)
        anomaly_diff = intensity_anomaly_diff 

        img_thr = org_image.astype(np.float32) * input_mask 
        anomaly_thr = anomaly_source_img_augmented.astype(np.float32) * (1-input_mask)

        synthetic_image = img_thr + anomaly_thr 


        no_anomaly = torch.rand(1).numpy()[0]
        # generate synthetic defect image for training a network 
        if self.for_check ==False:
            if no_anomaly > self.aug_ratio or defect_seg_nums==1:
                image = image.astype(np.float32)
                return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            else:
                if self.loss_type == 'smooth_cls':
                    input_mask_thr = 1-input_mask
                    diff = torch.FloatTensor(anomaly_diff)
                    diff[diff==0] = -100
                    smooth_msk = torch.sigmoid(diff*6)
                    smooth_msk[diff==-100] =0
                    smooth_msk = smooth_msk.detach().numpy() 
                    smooth_msk_expanded = np.concatenate([copy.deepcopy(np.expand_dims(smooth_msk, axis=0)) for i in range(1, defect_seg_nums)], axis=0)
                    defect_labels_expanded = np.concatenate([np.uint8(np.expand_dims(np.expand_dims(defect_seg_labels==i, axis=-1), axis=0)) for i in range(1, defect_seg_nums)], axis=0)
                    max_per_segments = np.max(np.multiply(smooth_msk_expanded, defect_labels_expanded), axis = (1,2), keepdims=True)
                    seg_smooth_msk = np.where(defect_labels_expanded==1, max_per_segments, 0)
                    msk = np.mean(np.sum(seg_smooth_msk, axis=0), axis=-1, keepdims =True)
                    msk = msk*input_mask_thr
                    msk[msk<=self.anomaly_confidence] = 0
                else:
                    msk = np.expand_dims(1-input_mask_org.astype(np.float32)/255.0, axis=-1)
                    msk[msk>self.anomaly_confidence] = 1.0
                    msk[msk<=self.anomaly_confidence] = 0
                has_anomaly = 1.0
                if np.sum(msk) == 0:
                    has_anomaly=0.0
                    msk = np.zeros_like(msk)
                    synthetic_image = org_image
                else:
                    binary_msk = copy.deepcopy(msk)
#                    binary_msk[binary_msk>0]=1.0
                    synthetic_image = synthetic_image * binary_msk + org_image.astype(np.float32)*(1-binary_msk) 
                return synthetic_image, msk, np.array([has_anomaly],dtype=np.float32)
        # generate synthetic defect image for saving samples  
        else:
            if no_anomaly > self.aug_ratio or defect_seg_nums==1:
                msk = np.zeros_like(perlin_thr, dtype=np.float32)
            else:
                if self.loss_type == 'smooth_cls':
                    input_mask_thr = 1-input_mask
                    diff = torch.FloatTensor(anomaly_diff)
                    diff[diff==0] = -100
                    smooth_msk = torch.sigmoid(diff*6)
                    smooth_msk[diff==-100] =0
                    smooth_msk = smooth_msk.detach().numpy() 
                    smooth_msk_expanded = np.concatenate([copy.deepcopy(np.expand_dims(smooth_msk, axis=0)) for i in range(1, defect_seg_nums)], axis=0)
                    defect_labels_expanded = np.concatenate([np.uint8(np.expand_dims(np.expand_dims(defect_seg_labels==i, axis=-1), axis=0)) for i in range(1, defect_seg_nums)], axis=0)
                    max_per_segments = np.max(np.multiply(smooth_msk_expanded, defect_labels_expanded), axis = (1,2), keepdims=True)
                    seg_smooth_msk = np.where(defect_labels_expanded==1, max_per_segments, 0)
                    msk = np.mean(np.sum(seg_smooth_msk, axis=0), axis=-1, keepdims =True)
                    msk = msk*input_mask_thr
                    msk[msk<=self.anomaly_confidence] = 0
                else:
                    msk = np.expand_dims(1-input_mask_org.astype(np.float32)/255.0, axis=-1)
                    msk[msk>self.anomaly_confidence] = 1.0
                    msk[msk<=self.anomaly_confidence] = 0
                has_anomaly = 1.0
                if np.sum(msk) == 0:
                    has_anomaly=0.0
                    msk = np.zeros(msk.shape)
                    synthetic_image = org_image
                else:
                    binary_msk = copy.deepcopy(msk)
#                    binary_msk[binary_msk>0]=1.0
                    synthetic_image = synthetic_image * binary_msk + org_image.astype(np.float32)*(1-binary_msk) 
            return org_image, anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, synthetic_image, msk 

    def transform_image(self, image, anomaly_source_path):
        if self.for_check ==False:
            synthetic_image, mask, has_anomaly = self.augment_image(image, anomaly_source_path)
            synthetic_image = np.transpose(synthetic_image, (2, 0, 1))
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            return synthetic_image, mask, has_anomaly
        else:
            org_image, anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, synthetic_image, msk = self.augment_image(image, anomaly_source_path)
            return org_image*255, anomalies_for_perlin_noise*255, perlin_thr*255, anomaly_source_img, anomaly_source_img_augmented*255, anomaly_mask*255, anomaly_samples*255, input_mask*255, img_thr*255, anomaly_thr*255, synthetic_image*255, msk*255

    def __getitem__(self, idx):
        image_org = cv2.imread(self.image_paths[idx])
        image_org = gray2rgb(image_org)
        image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
        if self.is_pairset==True:
            if image_org.shape[1] > image_org.shape[0]:
                image_org = cv2.resize(image_org, dsize=(self.resize_shape[1]*4, self.resize_shape[0]))
                image1 = image_org[:, :self.resize_shape[1], :]
                image2 = image_org[:, self.resize_shape[1]:self.resize_shape[1]*2, :]
            elif image_org.shape[0] > image_org.shape[1]:
                image_org = cv2.resize(image_org, dsize=(self.resize_shape[1], self.resize_shape[0]*4))
                image1 = image_org[:self.resize_shape[0], :, :]
                image2 = image_org[self.resize_shape[0]:self.resize_shape[0]*2, :, :]
            else:
                raise KeyboardInterrupt
        else:
            image = cv2.resize(image_org, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # generate the pair synthetic defect data
        if self.is_pairset==True:
            select_sample = torch.rand(1).numpy()[0]
            if select_sample < 0.5:
                if self.for_check ==False:
                    synthetic_image, mask, has_anomaly = self.get_anomaly_dataset(image1)
                    synthetic_image = torch.FloatTensor(synthetic_image)
                    synthetic_image = self.normalize(synthetic_image)
                    synthetic_image = synthetic_image.type(torch.float32)
                    mask = torch.FloatTensor(mask)
                    mask = mask.type(torch.float32)
                    image2 = torch.FloatTensor(np.transpose(image2, (2,0,1)))/255.0
                    image2 = self.normalize(image2)
                    image2 = image2.type(torch.float32)
                    return [synthetic_image, image2], has_anomaly, [mask, torch.zeros_like(mask).to(mask.device)]
                else:
                    _, anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, synthetic_image, mask = self.get_anomaly_dataset(image1) 
                    return [image1, image2], anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, [mask, np.zeros_like(mask)], [synthetic_image, image2] 
            else:
                if self.for_check ==False:
                    synthetic_image, mask, has_anomaly = self.get_anomaly_dataset(image2)
                    synthetic_image = torch.FloatTensor(synthetic_image)
                    synthetic_image = self.normalize(synthetic_image)
                    synthetic_image = synthetic_image.type(torch.float32)
                    mask = torch.FloatTensor(mask)
                    mask = mask.type(torch.float32)
                    image1 = torch.FloatTensor(np.transpose(image1, (2,0,1)))/255.0
                    image1 = self.normalize(image1)
                    image1 = image1.type(torch.float32)
                    return [image1, synthetic_image], has_anomaly, [torch.zeros_like(mask).to(mask.device), mask]
                else:
                    _, anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, synthetic_image, mask = self.get_anomaly_dataset(image2) 
                    return [image1, image2], anomalies_for_perlin_noise, perlin_thr, anomaly_source_img, anomaly_source_img_augmented, anomaly_mask, anomaly_samples, input_mask, img_thr, anomaly_thr, [np.zeros_like(mask), mask], [image1, synthetic_image] 
        # generate the single synthetic defect data
        else:
            if self.for_check ==False:
                synthetic_image, mask, has_anomaly = self.get_anomaly_dataset(image)
                synthetic_image = torch.FloatTensor(synthetic_image)
                synthetic_image = self.normalize(synthetic_image)
                synthetic_image = synthetic_image.type(torch.float32)
                mask = torch.FloatTensor(mask)
                mask = mask.type(torch.float32)
                return  synthetic_image, has_anomaly, mask
            else:
                return self.get_anomaly_dataset(image)

    # get images to blending with normal image to generate synthetic defects 
    def get_anomaly_dataset(self, input_image):
        anomaly_source_idx = torch.randint(0, len(self.image_paths), (1,)).item()
        if self.use_in_domain_data==0:
            return self.transform_image(input_image,'')
        else:
            in_domain = torch.rand(1).numpy()[0]
            if in_domain > 1-self.use_in_domain_data:
                anomaly_idx = torch.randint(0, len(self.image_paths), (1,)).item()
                return self.transform_image(input_image, self.image_paths[anomaly_idx])
            else:
                return self.transform_image(input_image, '')
