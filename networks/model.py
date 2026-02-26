'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import torch
import math
from torch import nn
from .resnet import *
from .visformer import *
from .resnet18_dec import *
from .visformer_tiny_dec import *
from .nf_arch import *
import timm


# forward hook to get input feature maps of NF networks 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# forward hook to get input feature maps of decoder networks 
dec_activation = {}
def get_dec_activation(name):
    def hook(model, input, output):
        dec_activation[name] = output.detach()
    return hook


# construct an encoder network 
def load_encoder_arch(c, L, model_cfg):
    pool_dims = list()
    nf_layers = ['layer'+str(i) for i in L]
    pool_cnt = 0
    if 'resnet' in c.network_arch:
        if  'resnet18' in c.network_arch:
            encoder = resnet18(c.is_pairset, L, c.att_layers, c.att_type, pretrained=c.pretrained, finetuning=c.finetuning, progress=True, concat_last_feat = model_cfg.backbone.concat_last_feat, nf_inp = c.nf_input)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.network_arch))

        for name, ch in encoder.named_modules():
            if isinstance(ch, nn.Conv2d)==True:
                if ch.padding[0]>0 and ch.padding[1]>0:
                    ch.padding_mode = 'reflect' 
                else:
                    pass
        for pool_cnt in range(len(L)):
            encoder.nf_skips[pool_cnt].register_forward_hook(get_activation(nf_layers[pool_cnt]))
            if 'twin' in c.train_type:
                encoder.dec_skips[pool_cnt].register_forward_hook(get_dec_activation(nf_layers[pool_cnt]))
            else:
                encoder.nf_skips[pool_cnt].register_forward_hook(get_dec_activation(nf_layers[pool_cnt]))
            if L[pool_cnt] ==0:
                out_channel = encoder.base_width
            else:
                out_channel = encoder.base_width * (2**(L[pool_cnt]-1))
            pool_dims.append(out_channel)
    elif 'visformer' in c.network_arch:
        if  'visformer_tiny' in c.network_arch:
            encoder = visformer_tiny(c.is_pairset, L, c.att_layers, c.att_type, pretrained=c.pretrained, finetuning=c.finetuning, concat_last_feat = model_cfg.backbone.concat_last_feat, nf_inp = c.nf_input)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.network_arch))

        for pool_cnt in range(len(L)):
            encoder.nf_skips[pool_cnt].register_forward_hook(get_activation(nf_layers[pool_cnt]))
            if 'twin' in c.train_type:
                encoder.dec_skips[pool_cnt].register_forward_hook(get_dec_activation(nf_layers[pool_cnt]))
            else:
                encoder.nf_skips[pool_cnt].register_forward_hook(get_dec_activation(nf_layers[pool_cnt]))
            if L[pool_cnt] ==0:
                out_channel = encoder.init_channels
            else:
                out_channel = encoder.embed_dim * (2**(L[pool_cnt]-1))
            pool_dims.append(out_channel)
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.network_arch))

    for name, ch in encoder.named_modules():
        if isinstance(ch, nn.Conv2d)==True:
            if ch.padding[0]>0 and ch.padding[1]>0:
                ch.padding_mode = 'reflect' 
            else:
                pass
    #
    return encoder, nf_layers, pool_dims


# construct a decoder network 
def load_decoder_arch(c, model_cfg, dec_dims):
    skip_layers_ = model_cfg.backbone.skip_layers

    if  'resnet18' in c.network_arch:
        if c.loss_type =='cls':
            dec = resnet18_dec(num_classes = c.num_class, skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, dec_dims = dec_dims, channel_reduction = model_cfg.backbone.dec_rate, concat_last_feat = model_cfg.backbone.concat_last_feat) 
        elif c.loss_type =='reg':
            dec = resnet18_dec(num_classes = c.img_dims[0], skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, dec_dims = dec_dims, channel_reduction = model_cfg.backbone.dec_rate, concat_last_feat = model_cfg.backbone.concat_last_feat) 
        elif 'smooth' in c.loss_type:
            dec = resnet18_dec(num_classes = c.num_class-1, skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, dec_dims = dec_dims, channel_reduction = model_cfg.backbone.dec_rate, concat_last_feat = model_cfg.backbone.concat_last_feat) 
        else:
            raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    elif  'visformer_tiny' in c.network_arch:
        if c.loss_type =='cls':
            dec = visformer_tiny_dec(num_classes = c.num_class, skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, in_channels = dec_dims[-1][0], channel_reduction = model_cfg.backbone.dec_rate, is_pairset = c.is_pairset) 
        elif c.loss_type =='reg':
            dec = visformer_tiny_dec(num_classes = c.img_dims[0], skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, in_channels = dec_dims[-1][0], channel_reduction = model_cfg.backbone.dec_rate, is_pairset = c.is_pairset) 
        elif 'smooth' in c.loss_type:
            dec = visformer_tiny_dec(num_classes = c.num_class-1, skip_layers = skip_layers_, final_activation ='sigmoid', skip_connection=model_cfg.backbone.skip_connection, in_channels = dec_dims[-1][0], channel_reduction = model_cfg.backbone.dec_rate, is_pairset = c.is_pairset) 
        else:
            raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.network_arch))

    for name, ch in dec.named_modules():
        if isinstance(ch, nn.Conv2d)==True:
            if ch.padding[0]>0 and ch.padding[1]>0:
                ch.padding_mode ='reflect' 
            elif isinstance(ch, nn.BatchNorm2d)==True:
                ch.track_running_stats =True
            else:
                pass

    dec_nf_layers = list()
    for l in range(max(skip_layers_)-1):
        dec_l = max(skip_layers_)-l-1
        dec_nf_layers.append(f"dec_layer{dec_l}")
    if 0 in skip_layers_:
        dec_nf_layers.append("dec_layer0")
    return dec, dec_nf_layers


# construct normalizing flow networks 
def load_nf_arch(c, model_cfg, dim_in, n_coupling_blocks):
    nf = freia_2dflow_head(dim_in, n_coupling_blocks, model_cfg.nf.kernel_size, model_cfg.nf.ch_rate, model_cfg.nf.relu_slope, model_cfg.nf.clamp_alpha, model_cfg.nf.linear_func)
    return nf
