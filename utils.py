'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, cv2, copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import os.path as osp
import sys
import tempfile
from importlib import import_module
from addict import Dict

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.detach().cpu().numpy() if tensor is not None else None

def np2t(np_array):
    '''numpy array -> pytorch tensor'''
    return torch.tensor(np.transpose(np_array, (2,0,1)), dtype = torch.float32)

# normalize to the range between 0 and 1
def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

# make directory if it does not exist 
def makedirs(dir_name):
    if os.path.isdir(dir_name)==False:
        os.makedirs(dir_name)

# convert a gray image to a rgb color image 
def gray2rgb(np_x):
    if len(np_x.shape)==2:  # handle greyscale classes
        x = np.expand_dims(np_x, axis=2)
        x = np.concatenate([x, x, x], axis=2)
    elif (len(np_x.shape)==3) and (np_x.shape[-1]==1):
        x = np_x
        x = np.concatenate([x, x, x], axis=2)
    else:
        x = np_x
        pass
    return x

def rgb2gray(np_x):
    return np.dot(np_x[...,:3], [0.2989, 0.5870, 0.1140])


def compare_models(model_1, model_2, txt_file=None, return_output = False, verbose =False):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                if verbose==True:
                    print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        if txt_file is None:
            pass
        else:
            txt_file.write('\nModels match perfectly! :)')
    if return_output==True:
        print(models_differ)
        return models_differ


# for measuring inference time
def time_measure():
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  
    return starter, ender


# write results in csv files
def write_csv(results:dict, csv_path, class_name):
    keys = list(results.keys())
    df_result = pd.DataFrame(results, index = [class_name])
    df_result.to_csv(csv_path, header=True, float_format='%.4f')


# calculate model size 
def cal_model_size_MB(model_list, result_dict):
    param_size = 0
    buffer_size = 0
    for model in model_list: 
        if isinstance(model, list):
            for model_ in model:
                for param in model_.parameters():
                    param_size += param.nelement() * param.element_size()
                for buffer in model_.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
        else:
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

    size_MB = (param_size + buffer_size) / 1024**2
    result_dict['model_size (MB)'] = size_MB
    print(f'model size: {size_MB:.3f} MB')


# make predictions with sizes of a full chip 
def make_full_chip(input_list, chip_list, file_paths, full_dims, H, W):
    outputs = []
    for k in range(len(input_list)):
        inputs_ = input_list[k]
        outputs_ = []
        if len(inputs_)==0:
            outputs_.append(input_list[k])
        elif len(inputs_)==len(file_paths):
            input_list[k] = [inputs_]
            num_channels = inputs_[0].shape[0]
            if num_channels ==3:
                output_ = np.zeros([len(chip_list),3]+full_dims)
            else:
                output_ = np.zeros([len(chip_list)]+full_dims)
            outputs_.append(output_)
        else:
            for i in range(len(inputs_)):
                num_channels = inputs_[i][0].shape[0]
                if num_channels ==3:
                    output_ = np.zeros([len(chip_list),3]+full_dims)
                else:
                    output_ = np.zeros([len(chip_list)]+full_dims)
                outputs_.append(output_)
        outputs.append(outputs_)
    
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        file_name_ = os.path.basename(file_path)
        file_name = file_name_.split('.')[0]
        chip_name  = '~'.join(file_name.split('~')[:-1])
        index_info = file_name.split('~')[-1] 
        idx_info_h = index_info.split('_')[-2]
        idx_info_w = index_info.split('_')[-1]
        init_h = int(idx_info_h.split('-')[-1])
        init_w = int(idx_info_w.split('-')[-1])

        fin_h=init_h + H
        fin_w=init_w + W
        
        chip_idx = chip_list.index(chip_name)
        
        for k in range(len(outputs)):
            outputs_ = outputs[k]
            for j in range(len(outputs_)):
                output_ = outputs_[j]
                if len(output_)==0:
                    pass
                else:
                    if output_[0].shape[0] ==3:
                        crop_out = output_[chip_idx][:, init_h:fin_h, init_w:fin_w]  
                        crop_out[crop_out!=0] = 0.5*(crop_out[crop_out!=0] + input_list[k][j][i][crop_out!=0])
                        crop_out[crop_out==0] = input_list[k][j][i][crop_out==0]
                        output_[chip_idx][:, init_h:fin_h, init_w:fin_w] = crop_out 
                    else:
                        crop_out = output_[chip_idx][init_h:fin_h, init_w:fin_w]  
                        crop_out[crop_out!=0] = 0.5*(crop_out[crop_out!=0] + input_list[k][j][i][crop_out!=0])
                        crop_out[crop_out==0] = input_list[k][j][i][crop_out==0]
                        output_[chip_idx][init_h:fin_h, init_w:fin_w] = crop_out 
    return_list = []
    for k in range(len(input_list)):
        inputs_ = input_list[k]
        if len(inputs_)==0:
            outputs_ = inputs_ 
        elif len(inputs_)==1:
            outputs_ = outputs[k][0]
        else:
            outputs_ = outputs[k]
        return_list.append(outputs_)
    return return_list 


def crop_full_chip(input_list, chip_list, file_paths, full_dims, H, W):
    return_list = []
    for k in range(len(input_list)):
        input_ = input_list[k] 
        output_ = []
        if len(input_)!= len(chip_list):
            output_ = [[] for i in range(len(input_))]
        elif len(input_)==0:
            return_list.append(output_)
            continue
        else:
            pass
        for i in range(len(file_paths)):
            file_path = file_paths[i]
            file_name_ = os.path.basename(file_path)
            file_name = file_name_.split('.')[0]
            chip_name  = '~'.join(file_name.split('~')[:-1])
            index_info = file_name.split('~')[-1] 
            idx_info_h = index_info.split('_')[-2]
            idx_info_w = index_info.split('_')[-1]
            init_h = int(idx_info_h.split('-')[-1])
            init_w = int(idx_info_w.split('-')[-1])

            fin_h=init_h + H
            fin_w=init_w + W

            chip_idx = chip_list.index(chip_name)
        
            if len(input_)!= len(chip_list):
                for i in range(len(input_)):
                    if len(input_[i][chip_idx].shape)==3:
                        crop_out = input_[i][chip_idx][:, init_h:fin_h, init_w:fin_w] 
                    else:
                        crop_out = input_[i][chip_idx][init_h:fin_h, init_w:fin_w] 
                    output_[i].append(np.expand_dims(crop_out, axis=0))
            else:
                if len(input_[chip_idx].shape)==3:
                    crop_out = input_[chip_idx][:, init_h:fin_h, init_w:fin_w] 
                else:
                    crop_out = input_[chip_idx][init_h:fin_h, init_w:fin_w] 

                output_.append(np.expand_dims(crop_out, axis=0))
        if len(input_)!= len(chip_list):
            for i in range(len(input_)):
                output_[i] = np.concatenate(output_[i], axis=0)
        else:
            output_ = np.concatenate(output_, axis=0)
        return_list.append(output_)
    return return_list


# make a path to save models according to configures 
def make_model_path(c, model_cfg):
    if c.finetuning==False:
        if c.nf_aug_ratio_train==0:
            data_info = 'real_pretrained'
        else:
            data_info = f'synthetic{int(c.nf_aug_ratio_train*100):03d}_pretrained'
    else:   
        data_info = f'synthetic{int(c.aug_ratio_train*100):03d}'

    if (('nf_only' in c.train_type)==True or ('fe_only' in c.train_type)==True) and c.finetuning==True:
        if 'single' in c.train_type:
            train_method = 'single'
        else:
            train_method = 'twin'
    else:
        train_method = c.train_type 
    
    if len(c.att_layers)>0:
        al_str = [str(al) for al in c.att_layers]
        att_layers = '-'.join(al_str)
        if len(c.att_type)==0:
            raise KeyboardInterrupt
        train_method = f'{train_method}_{c.att_type}-al{att_layers}'
    else:
        pass

    if c.finetuning ==True:
        train_method = f'{train_method}-{c.loss_type}'
    else:
        pass

    
    c.model_name = "{}-{}-{}".format(data_info, train_method, model_cfg.backbone.type)

    model_dir = os.path.join(c.model_path, c.data_settings.split('_')[0], f'inp_{c.input_size}', c.model_name, f'run_{c.run_name}', f're_{c.repeatability}')
    # model_dir = os.path.join(c.model_path, c.data_settings.split('_')[0], f'inp_{c.input_size}', c.model_name, f'run_{c.run_name}', f'seed_{c.seed}')
    return model_dir


class cfg_dict(Dict):
    '''
        모델 cfg dictionary에 해당 key, value가 있는지 확인

        -Method
            __missing__: dict에 해당 key 값이 없으면 KeyError가 뜸
                -Arguments
                    name: dict key 값
            __getattr__: 객체의 속성 가져올 때 사용  
                -Arguments
                    name: dict key 값
                -Return
                    value: key 값 존재할 때 value 값 
    '''
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(cfg_dict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class cfg(object):
    '''
        {cfg_dir}/{filename}.py의 모델 cfg 파일 내용을 dictionary로 변환    
    '''
    @staticmethod
    def _file2dict(filename):
        ''' 
            cfg file을 dictionary로 변환
        '''
        filename = str(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_cfg_dir:
                shutil.copyfile(filename,
                                osp.join(temp_cfg_dir, '_tempcfg.py'))
                sys.path.insert(0, temp_cfg_dir)
                mod = import_module('_tempcfg')
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules['_tempcfg']
        else:
            raise IOError('Only .py type are supported now!')
        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename):
        '''
            cfg file을 dictionary로 변환한 후 dictionary 읽기
        '''
        cfg_dict_, cfg_text = cfg._file2dict(filename)
        return cfg(cfg_dict_, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict_=None, cfg_text=None, filename=None):
        if cfg_dict_ is None:
            cfg_dict_ = dict()
        elif not isinstance(cfg_dict_, dict):
            raise TypeError('cfg_dict_ must be a dict, but got {}'.format(
                type(cfg_dict_)))

        super(cfg, self).__setattr__('_cfg_dict', cfg_dict(cfg_dict_))
        super(cfg, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(cfg, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'cfg (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = cfg_dict(value)
        self._cfg_dict.__setattr__(name, value)


def str_table(result, width):
    space = (width-len(result)-1)//(len(result))
    last_space = width-len(result)-1-space*(len(result)-1)
    spaces = [space]*(len(result)-1)+[last_space] 

    result_str = '|'
    for i in range(len(result)):
        result_str += f'{result[i]}'.center(spaces[i])
        result_str += '|'
    return result_str
