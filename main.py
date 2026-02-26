'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
from __future__ import print_function
import os, random, time, math, datetime
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from args import get_args
from utils import * 
from train_methods import *
from custom_datasets import * 
import shutil
import socket


# for repeatability 
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

# cache the input configure values
def cache_test_args(c, config_list):
    temp_config_list = []
    for arg_name in config_list:
        val = getattr(c, arg_name)
        temp_config_list.append(val)
    return temp_config_list

# replace the values saved in the state dict of the trained model to the input configures for inference
def redefine_test_args(c, config_list, temp_config_list):
    for i in range(len(config_list)):
        arg_name = config_list[i]
        setattr(c, arg_name, temp_config_list[i])

def main(c):
    data_cfg = cfg.fromfile(os.path.join(c.data_cfg_dir, f'{c.data_settings}.py'))
    model_cfg = cfg.fromfile(os.path.join(c.model_cfg_dir, f'{c.network_arch}.py')) 

    if type(c.nf_input)==list:
        c.nf_input_str = '-'.join(c.nf_input)
    else:
        c.nf_input_str = c.nf_input

    if c.is_train==True:
        c.model_dir = make_model_path(c, model_cfg)
        makedirs(c.model_dir)
        if 'nf_only' in c.train_type:
            nl_str = [str(nl) for nl in c.nf_layers]
            nf_layers = '-'.join(nl_str)
            if len(c.att_type)>0:
                c.train_type = f'{c.train_type}_aug{c.nf_aug_ratio_train}_inp{c.nf_input_str}_{model_cfg.nf.type}_nl{nf_layers}'
            else:
                c.train_type = f'{c.train_type}_aug{c.nf_aug_ratio_train}_{model_cfg.nf.type}_nl{nf_layers}'

        if os.path.isdir(f'{c.model_dir}/codes')==True:
            shutil.rmtree(f'{c.model_dir}/codes')
        shutil.copytree('./',f'{c.model_dir}/codes')
        c.run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    else:
        path=os.path.dirname(os.path.abspath(__file__))
        if 'models' in path:
            codes_dir_name = path.split(os.sep)[-1]
            model_dir=path.replace(f'/{codes_dir_name}', '')
            print(model_dir)
        else:
            c.model_dir = make_model_path(c, model_cfg)
            weight_dir = os.path.join(c.model_dir, 'weights')
            if ('fe_only' in c.infer_type)==True:
                sub_path = c.infer_type
            else:
                nl_str = [str(nl) for nl in c.nf_layers]
                nf_layers = '-'.join(nl_str)
                if 'joint' in c.infer_type:
                    infer_type = c.infer_type.replace('joint', 'nf_only')
                else:
                    infer_type = c.infer_type 

                if len(c.att_type)>0:
                    sub_path = f'{infer_type}_aug{c.nf_aug_ratio_train}_inp{c.nf_input_str}_{model_cfg.nf.type}_nl{nf_layers}'
                else:
                    sub_path = f'{infer_type}_aug{c.nf_aug_ratio_train}_{model_cfg.nf.type}_nl{nf_layers}'

            model_files_=os.listdir(weight_dir)
            for model_file_ in model_files_:
                if (sub_path in model_file_)==True:
                    model_path=os.path.join(weight_dir, model_file_)
                else: 
                    pass

            if 'model_path' not in locals():
                print(f'There is no trained files (train type: {sub_path})')
                raise KeyboardInterrupt

        # configure list to replace values saved in the state_dict of trained model to input values for inference
        config_list = ['batch_size', 'workers', 'train_type', 'is_close', 'is_open', 'is_k_disk', 'feat_avg_topk', 'k_size', 'th_pix', 'data_path', 'aug_data_path', 'data_settings', 'th_manual', 'infer_type', 'is_train', 'test_data_type', 'model_dir', 'pro', 'viz', 'w_fe', 'get_best_w_fe', 'save_all_tn_sample', 'is_full', 'save_features', 'fn', 'viz_total_features', 'viz_diff_features', 'repeat_num',  'init_chip', 'fin_chip', 'nf_input']

        if c.seed == None:
            pass
        else:
            config_list.extend(['seed'])

        temp_config_list = cache_test_args(c, config_list)
        state = torch.load(model_path)
        c = state['args']

        redefine_test_args(c, config_list, temp_config_list)
        del state

        if 'joint' in c.infer_type:
            c.infer_type = sub_path.replace('nf_only', 'joint')
        else:
            c.infer_type = sub_path

    # image
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    c.img_dims = [3] + list(c.img_size)
    c.train_data_path = os.path.join(c.data_path, data_cfg.product.train, data_cfg.layer_type.train, data_cfg.data_type.train)
    c.train_aug_data_path = os.path.join(c.aug_data_path, data_cfg.product.train, data_cfg.layer_type.train, data_cfg.data_type.train)
    c.test_aug_data_path = os.path.join(c.aug_data_path, data_cfg.product.test_aug, data_cfg.layer_type.test_aug, data_cfg.data_type.test_aug)
    c.test_data_path = os.path.join(c.data_path, data_cfg.product.test, data_cfg.layer_type.test, data_cfg.data_type.test)

    num_class =1 
    if c.aug_ratio_train >0:
        num_class += 1 
    else:
        pass

    c.num_class = num_class

    # set hyper parameters for scheduling learning rate 
    if 'fe_only' in c.train_type:
        total_epoch = c.meta_epochs+c.freeze_enc_epochs
        c.lr_decay_epochs = [int(c.lr_decay_epochs_percentage[i]*total_epoch) for i in range(len(c.lr_decay_epochs_percentage))]
        print('LR schedule: {}'.format(c.lr_decay_epochs))
        if c.lr_warm:
            c.lr_warmup_from = c.lr/10.0
            if c.lr_cosine:
                eta_min = c.lr * (c.lr_decay_rate ** 3)
                c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                        1 + math.cos(math.pi * c.lr_warm_epochs / total_epoch)) / 2
            else:
                c.lr_warmup_to = c.lr
    else:
        total_epoch = c.meta_epochs
        # milestones 그대로 계산
        c.lr_decay_epochs = [int(p * total_epoch) for p in c.lr_decay_epochs_percentage]
        print(f"LR schedule (epochs): {c.lr_decay_epochs}")

        # === base LR 리스트 만들기 ===
        # 우선순위: lr_nf_list 있으면 그것 사용, 없으면 단일 c.lr을 NF 수만큼 복제
        num_groups = len(getattr(c, 'nf_layers', [])) or 1
        if getattr(c, 'lr_nf_list', None) is not None:
            assert len(c.lr_nf_list) == num_groups, \
                f"--lr_nf_list 길이({len(c.lr_nf_list)}) != nf_layers({num_groups})"
            base_lrs = [float(x) for x in c.lr_nf_list]
        else:
            base_lrs = [float(c.lr)] * num_groups

        if c.lr_warm:
            # from: base_lr * 0.1
            c.lr_warmup_from_list = [lr / 10.0 for lr in base_lrs]

            if c.lr_cosine:
                # eta_min_i = base_lr_i * (lr_decay_rate ** 3)
                decay_pow = c.lr_decay_rate ** 3
                cos_factor = (1 + math.cos(math.pi * c.lr_warm_epochs / total_epoch)) / 2.0
                # to: eta_min_i + (base_lr_i - eta_min_i) * cos_factor
                c.lr_warmup_to_list = [
                    (lr * decay_pow) + (lr - lr * decay_pow) * cos_factor
                    for lr in base_lrs
                ]
            else:
                # cosine 아니면 to = base_lr
                c.lr_warmup_to_list = base_lrs[:]

            print("warmup_from_list:", c.lr_warmup_from_list)
            print("warmup_to_list  :", c.lr_warmup_to_list)

    # set device 
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    if c.seed ==None:
        c.seed = int(time.time())
    init_seeds(seed=c.seed)
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    c.device = torch.device(f"cuda:{c.gpu}" if c.use_cuda else "cpu")

    c_dict = vars(c)
    dict_string=''
    for k in c_dict.keys():
        dict_string = f'{dict_string}\n{k}:{c_dict[k]}'
    print(dict_string)

    # train or test  
    if c.is_train ==True:
        # save configure file
        with open(os.path.join(c.model_dir, f'{c.train_type}_config_list.txt'), 'w') as f:
            f.write(dict_string)

        if 'single' in c.train_type:
            c.is_pairset = False
            if c.finetuning == False:
                log_txt_path = train_NFs_pretrain.run(c, data_cfg, model_cfg)
            elif 'fe_only' in c.train_type:
                log_txt_path = finetune_enc_single.run(c, data_cfg, model_cfg)
            elif 'nf_only' in c.train_type:
                log_txt_path = train_NFs_finetune.run(c, data_cfg, model_cfg)
            else:
                raise NotImplementedError(f'{c.train_type} is not implemented for train_type.')
        elif 'twin' in c.train_type:
            c.is_pairset = True
            if c.finetuning == False:
                log_txt_path = train_diff_NFs_pretrain.run(c, data_cfg, model_cfg)
            elif 'fe_only' in c.train_type:
                log_txt_path = finetune_enc_twin.run(c, data_cfg, model_cfg)
            elif 'nf_only' in c.train_type:
                log_txt_path = train_diff_NFs_finetune.run(c, data_cfg, model_cfg)
            else:
                raise NotImplementedError(f'{c.train_type} is not implemented for train_type.')
        else:
            raise NotImplementedError(f'{c.train_type} is not implemented for train_type.')
    else:
        if 'joint' in c.infer_type:
            c.add_fe_anomaly =True
            assert c.w_fe>0 or c.get_best_w_fe==True
        else:
            c.add_fe_anomaly =False	

        if 'single' in c.infer_type:
            c.is_pairset = False
            assert c.viz_diff_features ==False
            assert c.save_features ==False 
            if c.finetuning == False:
                log_txt_path = train_NFs_pretrain.run(c, data_cfg, model_cfg)
            elif 'fe_only' in c.infer_type:
                log_txt_path = finetune_enc_single.run(c, data_cfg, model_cfg)
            elif ('nf_only' in c.infer_type)==True or ('joint' in c.infer_type)==True:
                log_txt_path = train_NFs_finetune.run(c, data_cfg, model_cfg)
            else:
                raise NotImplementedError(f'{c.infer_type} is not implemented for infer_type.')
        elif 'twin' in c.infer_type:
            c.is_pairset = True
            if c.finetuning == False:
                log_txt_path = train_diff_NFs_pretrain.run(c, data_cfg, model_cfg)
            elif 'fe_only' in c.infer_type:
                assert c.save_features ==False 
                log_txt_path = finetune_enc_twin.run(c, data_cfg, model_cfg)
            elif ('nf_only' in c.infer_type)==True or ('joint' in c.infer_type)==True:
                log_txt_path = train_diff_NFs_finetune.run(c, data_cfg, model_cfg)
            else:
                raise NotImplementedError(f'{c.infer_type} is not implemented for infer_type.')
        else:
            raise NotImplementedError(f'{c.infer_type} is not implemented for infer_type.')

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip_address = s.getsockname()[0]
    s.close()
    c.com_num = ip_address.split('.')[-1]
    if c.is_train==False:
        txt_file = open(log_txt_path, 'r')
        log_str = txt_file.read()
        txt_file.close()
        if len(c.att_type)>0:
            print(f'[{c.data_settings.upper()}, {c.com_num}] {c.network_arch}-{c.att_type}-{c.infer_type} inference process is finish!', dict_string+'\n'+log_str) 
        else:
            print(f'[{c.data_settings.upper()}, {c.com_num}] {c.network_arch}-{c.infer_type} inference process is finish!', dict_string+'\n'+log_str) 
    else:
        log_str = ''
        if len(c.att_type)>0:
            print(f'[{c.data_settings.upper()}, {c.com_num}] {c.network_arch}-{c.att_type}-{c.train_type} training process is finish!', dict_string+'\n'+log_str) 
        else:
            print(f'[{c.data_settings.upper()}, {c.com_num}] {c.network_arch}-{c.train_type} training process is finish!', dict_string+'\n'+log_str) 


if __name__ == '__main__':
    c = get_args()
    main(c)

