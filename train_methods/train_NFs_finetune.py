'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from .evaluate import *
from utils import *
from custom_datasets import *
from networks import *


# train 2d-Flow with the finetuned feature extractor
def train_meta_epoch(c, model_cfg, epoch, loader, model, optimizers, pool_layers_names, txt_file):
    log_theta = torch.nn.LogSigmoid()
    I = len(loader)
    iterator = iter(loader)

    encoder, _, nfs= model
    encoder = encoder.eval()
    nfs = [nf.train() for nf in nfs]

    # update learning rate
    adjust_group_learning_rate(c, optimizers, epoch, c.meta_epochs)

    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_defect_nf_loss = 0.0
        train_good_nf_loss = 0.0
        train_count = 0
        train_defect_nf_count = 0
        train_good_nf_count = 0
        nf_losses = []
        for i in range(I):
            # warm-up learning rate
            lrs = warmup_group_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizers)
            # sample batch
            try:
                image, _, mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, mask = next(iterator)

            if mask.size(1)>1:
                mask = torch.mean(mask,1, keepdim=True)
            else:
                pass
            mask = (mask>0).type(torch.long)

            # encoder prediction
            image = image.to(c.device)  # single scale
            mask = mask.to(c.device)
            with torch.no_grad():
                _ = encoder(image)

            # train nf
            for pl, l in enumerate(c.nf_layers):
                l_idx = model_cfg.backbone.skip_layers.index(l)
                layer = pool_layers_names[l_idx]

                e = activation[layer].detach()  # BxCxHxW
                    
                B, C, H, W = e.size()

                # ground truth for feature maps
                mask = mask.type(torch.float32)
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m[m>0]=-1
                m[m!=-1]=1

                # train a 2d-Flow network
                nf = nfs[pl]
                z, log_jac_det= nf(e)

                # get loss 
                nf_log_prob, defect_logp, good_logp = get_logp_2d(C, z, log_jac_det[0], m, True)
                log_prob = nf_log_prob / C  # likelihood per dim
                #
                train_defect_nf_loss += defect_logp.sum() /C
                train_defect_nf_count += torch.sum(m==-1)
                train_good_nf_loss += good_logp.sum() /C
                train_good_nf_count += torch.sum(m==1)
                loss = -log_theta(log_prob)

                opt = optimizers[l_idx]
                opt.zero_grad()
                loss.mean().backward()
                opt.step()

                if len(nf_losses) < len(c.nf_layers):
                    nf_losses.append(loss.mean()*B)
                else:
                    nf_losses[pl] += loss.mean()*B
                train_loss += t2np(loss.mean()*B)
                train_count += B

        # show results
        mean_train_total_loss = train_loss / train_count
        mean_train_defect_nf_loss = train_defect_nf_loss / train_defect_nf_count
        mean_train_good_nf_loss = train_good_nf_loss / train_good_nf_count
        loss_str = ''
        for l in range(len(nf_losses)-1):
            loss_str += f'nf-{c.nf_layers[l]}_loss: {nf_losses[l]/train_count:.4f}, '
        loss_str += f'nf-{c.nf_layers[len(nf_losses)-1]}_loss: {nf_losses[len(nf_losses)-1]/train_count:.4f}'
        
        # nf_layers와 길이 맞춰서 포맷
        nf_cnt = min(len(lrs), len(c.nf_layers))
        lr_str = ', '.join([f'nf-{c.nf_layers[i]}_lr:{lrs[i]:.2g}' for i in range(nf_cnt)])

        # --- 로그 출력/저장 ---
        if c.verbose:
            print(f'Epoch: {epoch:02d}.{sub_epoch:03d}\t'
                f'train_nf_loss: {mean_train_total_loss:.4f}, '
                f'train_defect_nf_loss: {mean_train_defect_nf_loss:.4f}, '
                f'train_good_nf_loss: {mean_train_good_nf_loss:.4f}')
            print(loss_str)
            print(f'{lr_str}')

        txt_file.write(f'\nEpoch: {epoch:02d}.{sub_epoch:03d}\t'
                    f'train_nf_loss: {mean_train_total_loss:.4f}, '
                    f'train_defect_nf_loss: {mean_train_defect_nf_loss:.4f}, '
                    f'train_good_nf_loss: {mean_train_good_nf_loss:.4f}')
        txt_file.write(f'\n{loss_str}')
        txt_file.write(f'\n{lr_str}')


def test_meta_epoch(c, model_cfg, epoch, loader, model, pool_layers_names, fe_loss_fn, txt_file):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    txt_file.write('\nCompute loss and scores on test set:')

    encoder, decoder,  nfs = model
    encoder = encoder.eval()
    decoder = decoder.eval()
    nfs = [nf.eval() for nf in nfs]
    feature_maps = []

    log_theta = torch.nn.LogSigmoid()
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    if 'cls' in c.loss_type:
        pred_list = list()
    elif c.loss_type =='reg':
        pred_list = [list(), list()]
    else:
        raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    test_dist = [list() for layer in c.nf_layers]

    test_defect_nf_loss = 0.0
    test_defect_nf_count = 0
    test_good_nf_loss = 0.0
    test_good_nf_count = 0
    test_loss = 0.0
    test_count = 0
    nf_losses = []

    with torch.no_grad():
        for i, (image, label, gt_mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(np.float32(t2np(gt_mask)))
            if gt_mask.size(1)>1:
                mask = torch.mean(gt_mask,1, keepdim=True)
            else:
                mask = gt_mask
            mask = (mask>0).type(torch.long)

            # data
            image = image.to(c.device) # single scale
            mask = mask.to(c.device)

            # inference enc-dec
            enc_out = encoder(image)
            if model_cfg.backbone.skip_connection ==True:
                pix_out = decoder(enc_out)
            else:
                pix_out = decoder(enc_out[-1])

            if c.loss_type =='cls':
                pix_loss = fe_loss_fn(pix_out, torch.squeeze(mask,1))
                pix_loss = torch.unsqueeze(pix_loss,1)
                prob_map = torch.softmax(pix_out, 1)
                pred_list.extend(t2np(1-prob_map[:,0,:,:]))
            elif c.loss_type =='reg':
                pix_loss = torch.mean(fe_loss_fn(pix_out, image), 1, keepdim=True)
                pred_list[0].extend(t2np(pix_out))
                pred_list[1].extend(t2np(torch.squeeze(pix_loss,1)))
            elif 'smooth' in c.loss_type:
                pix_loss = fe_loss_fn(pix_out, mask.type(torch.float))
                pred_list.extend(t2np(torch.squeeze(pix_out,1)))
            else:
                raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))

            # save feature maps
            if c.is_train ==False and c.viz_total_features ==True:
                f_num = 0 
                for pl, l in enumerate(c.nf_layers):
                    l_idx = model_cfg.backbone.skip_layers.index(l)
                    layer = pool_layers_names[l_idx]

                    e = activation[layer].detach()  # bxcxhxw

                    _, c_idx = torch.topk(torch.mean(e, (-2,-1)), int(c.feat_avg_topk*e.size(1)), 1)

                    feat_map_sorted = [] 
                    for b in range(c_idx.size(0)):
                        feat_map_sorted.append(e[b,c_idx[b],:,:])
                    feat_map_sorted_mean = torch.unsqueeze(torch.mean(torch.stack(feat_map_sorted,0), 1), 1)
                    feat_map = F.interpolate(feat_map_sorted_mean, size=(image.size(-2), image.size(-1)), mode = 'bicubic', align_corners =True)
                    if i==0:
                        feature_maps.append([])
                        feature_maps[f_num].extend(t2np(torch.squeeze(feat_map, 1)))
                    else:
                        feature_maps[f_num].extend(t2np(torch.squeeze(feat_map, 1)))
                    f_num += 1


            # inference NFs
            for pl, l in enumerate(c.nf_layers):
                l_idx = model_cfg.backbone.skip_layers.index(l)
                layer = pool_layers_names[l_idx]

                e = activation[layer].detach()  # BxCxHxW

                B, C, H, W = e.size()

                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)

                # ground truth for feature maps
                mask = mask.type(torch.float32)
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m[m>0]=-1
                m[m!=-1]=1

                # inference of a 2d-Flow network
                nf = nfs[pl]
                z, log_jac_det= nf(e)

                # get loss
                nf_log_prob, defect_logp, good_logp= get_logp_2d(C, z, log_jac_det[0], m, False)

                test_defect_nf_loss += defect_logp.sum() /C
                test_defect_nf_count += torch.sum(m==-1)
                test_good_nf_loss += good_logp.sum() /C
                test_good_nf_count += torch.sum(m==1)
                log_prob = nf_log_prob / C  # likelihood per dim

                # save log-likelihood
                test_dist[pl] = test_dist[pl] + log_prob.detach().cpu().tolist()

                loss = -log_theta(log_prob)

                if len(nf_losses) < len(c.nf_layers):
                    nf_losses.append(loss.mean()*B)
                else:
                    nf_losses[pl] += loss.mean()*B
                test_loss += t2np(loss.mean()*B)
                test_count += B

    # show results
    mean_test_loss = test_loss / test_count
    mean_test_defect_nf_loss = test_defect_nf_loss / test_defect_nf_count
    mean_test_good_nf_loss = test_good_nf_loss / test_good_nf_count
    loss_str = ''
    for l in range(len(nf_losses)-1):
        loss_str += f'nf-{c.nf_layers[l]}_loss: {nf_losses[l]/test_count:.4f}, '
    loss_str += f'nf-{c.nf_layers[len(nf_losses)-1]}_loss: {nf_losses[len(nf_losses)-1]/test_count:.4f}'

    if c.verbose:
        print(f'Epoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
        print(loss_str)
    txt_file.write(f'\nEpoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
    txt_file.write(f'\n{loss_str}')
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list, feature_maps, pred_list


def test_meta_fps(c, model_cfg, epoch, loader, model, pool_layers_names, fe_loss_fn, txt_file, result_dict):
    if c.verbose:
        print('\nCompute inference speed (fps) on test set:')
    txt_file.write('\nCompute inference speed (fps) on test set:')

    encoder, decoder,  nfs = model
    encoder = encoder.eval()
    if c.add_fe_anomaly ==True:
        decoder = decoder.eval()
    else:
        del decoder
    nfs = [nf.eval() for nf in nfs]

    starter, ender = time_measure()
    with torch.no_grad():
        # warm-up
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            image = image.to(c.device) 
            enc_out = encoder(image)  
            if c.add_fe_anomaly==True:
                if model_cfg.backbone.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])
            else:
                pass
        
        # time measure
        starter.record()
        for k in range(c.repeat_num):
            for i, (image, _, gt_mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                # data
                if gt_mask.size(1)>1:
                    mask = torch.mean(gt_mask,1, keepdim=True)
                else:
                    mask = gt_mask
                mask = (mask>0).type(torch.long)

                image = image.to(c.device) # single scale
                mask = mask.to(c.device)

                # enc-dec inference
                enc_out = encoder(image)

                if c.add_fe_anomaly==True:
                    if model_cfg.backbone.skip_connection ==True:
                        pix_out = decoder(enc_out)
                    else:
                        pix_out = decoder(enc_out[-1])
                else:
                    pass

                # 2d-Flow inference
                for pl, l in enumerate(c.nf_layers):
                    l_idx = model_cfg.backbone.skip_layers.index(l)
                    layer = pool_layers_names[l_idx]

                    e = activation[layer]  # BxCxHxW

                    B, C, H, W = e.size()

                    #
                    mask = mask.type(torch.float32)
                    m = F.interpolate(mask, size=(H, W), mode='nearest')
                    m[m>0]=-1
                    m[m!=-1]=1
                    #
                    nf = nfs[pl]
                    z, log_jac_det = nf(e)

                    nf_log_prob, defect_logp, good_logp= get_logp_2d(C, z, log_jac_det[0], m, False)
        ender.record()
        torch.cuda.synchronize()
        speed_result = starter.elapsed_time(ender)

    # show results
    fps = c.repeat_num*len(loader.dataset) / (speed_result/1000)
    if c.verbose:
        print(f'Batch size: {c.batch_size}, Repeat num: {c.repeat_num}, Inference time: {speed_result:0.2f}msec,  Data num: {len(loader.dataset)}, fps: {fps:.2f} fps')
    txt_file.write(f'\nBatch size: {c.batch_size}, Repeat num: {c.repeat_num}, Inference time: {speed_result:0.2f}msec,  Data num: {len(loader.dataset)}, fps: {fps:.2f} fps')
    result_dict['batch_size'] = c.batch_size
    result_dict['repeat_num'] = c.repeat_num
    result_dict['inference_time (msec)'] = speed_result
    result_dict['data_num'] = len(loader.dataset)
    result_dict['fps'] = fps


def run(c, data_cfg, model_cfg):
    if c.is_train ==True:
        # log
        log_txt_path = os.path.join(c.model_dir, f'{c.train_type}_train_progress.txt')
        model_file = open(os.path.join(c.model_dir, f'{c.train_type}_model_summary.txt'), 'w')
    else: 
        # make a path of directory to save test results
        dir_feature=[]
        if c.test_data_type == 'aug':
            dir_feature.append(f'aug')
        if c.th_manual>0:
            dir_feature.append(f'manual_th_{c.th_manual:0.4f}')
        if c.fn >=0:
            dir_feature.append(f'fixed_fn_{c.fn}')
        if c.is_open==True:
            dir_feature.append('imopen')
        if c.is_close==True:
            dir_feature.append('imclose')

        if c.add_fe_anomaly ==True:
            if c.get_best_w_fe==True:
                map_type = f'{c.infer_type}-find_best_w_fe'
            else:
                map_type = f'{c.infer_type}-{c.w_fe:0.2f}'
        else:
            map_type = c.infer_type

        if len(dir_feature)>0:
            str_dir_features = '-'.join(dir_feature)
            tag = os.path.join(map_type, str_dir_features)
        else:
            tag = os.path.join(map_type, 'classic')
        save_dir = os.path.join(c.model_dir, c.data_settings, data_cfg.class_name.test, tag)
        makedirs(save_dir)
        log_txt_path = os.path.join(save_dir, f'{map_type}_test_results.txt')
    txt_file = open(log_txt_path, 'a')


    #============================================
    #               Set Model
    #============================================

    # set the finetuned encoder
    encoder, pool_layers_names, pool_dims_total = load_encoder_arch(c, model_cfg.backbone.skip_layers, model_cfg)
    encoder = encoder.eval()
    if c.is_train ==True:
        sample = torch.zeros(tuple([2]+c.img_dims))
        _ = encoder(sample)
        model_file.write('Encoder Summary \n')
        model_file.write(str(encoder))
    else:
        pass
    encoder = encoder.to(c.device)

    # set decoder
    if c.is_train ==True:
        c.dec_dims_fe = []
        c.pool_layers_dec = []
        for l in range(len(model_cfg.backbone.skip_layers)):
            layer = pool_layers_names[l]
            c.pool_layers_dec.append(layer)
            e = dec_activation[layer].detach()  # BxCxHxW
            input_dims = [e.size(-3), e.size(-2), e.size(-1)]
            c.dec_dims_fe.append(input_dims)
    else:
        pass


    decoder, dec_pool_layers= load_decoder_arch(c, model_cfg, c.dec_dims_fe) 
    print(dec_pool_layers)
    if c.is_train ==True:
        model_file.write('Decoder Summary \n')
        if model_cfg.backbone.skip_connection ==True:
            model_file.write(str(decoder))
        else:
            dec_stat=summary(decoder,c.dec_dims_fe[-1], depth=5)
            model_file.write(str(dec_stat))
    else:
        pass
    decoder = decoder.to(c.device)

    model_loaded = [encoder, decoder] 
    load_weights(c, model_loaded, 'single_fe_only')


    # set NF
    nfs = []
    nf_param_groups = []
    optimizers = None
    indiv_pr_obs = {'exp-layerwise': [], 'exp_std-layerwise': [], 'gauss_cdf-layerwise': []}
    
    if c.is_train:
        # 학습: activation으로부터 dims 계산, NF 생성 + 개별 LR param group 구성
        c.nf_dims = []
        for i, nl in enumerate(c.nf_layers):
            l = model_cfg.backbone.skip_layers.index(nl)
            layer = pool_layers_names[l]
            n_coupling_blocks = model_cfg.nf.coupling_blocks[l]
            e = activation[layer].detach()  # BxCxHxW
            input_dims = [e.size(-3), e.size(-2), e.size(-1)]
            c.nf_dims.append(input_dims)
            print(n_coupling_blocks)
            nf_i = load_nf_arch(c, model_cfg, input_dims, n_coupling_blocks).to(c.device, non_blocking=True)
            nfs.append(nf_i)

        optimizers = [torch.optim.Adam(nf.parameters(), lr=lr_i) for nf, lr_i in zip(nfs, c.lr_nf_list)]
    else:
        for i, nl in enumerate(c.nf_layers):
            l = model_cfg.backbone.skip_layers.index(nl)
            dims = c.nf_dims[l]
            n_coupling_blocks = model_cfg.nf.coupling_blocks[l]
            nf_i = load_nf_arch(c, model_cfg, dims, n_coupling_blocks).to(c.device, non_blocking=True)
            nfs.append(nf_i)

    if c.is_train:
        # NF별 현재 학습률 출력
        for i, opt in enumerate(optimizers):
            lr = opt.param_groups[0]['lr']  # NF 하나당 optimizer 하나라 첫 그룹만 확인
            print(f"NF[{i}] lr={lr}")

        # NF 구조 출력 및 파일 기록
        for idx, nf in enumerate(nfs, 1):
            # print(f'NF({idx}) Structure \n{nf}')
            model_file.write(f'\n\nNF({idx}) Structure \n')
            model_file.write(str(nf) + '\n')
    
    model = [encoder, decoder, nfs]

    #============================================
    #               Set Data
    #============================================
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    # task data
    if c.nf_aug_ratio_train>0 and c.is_train ==True:
        train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, c.nf_aug_ratio_train, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, c.is_pairset, c.loss_type) 
        if c.repeat_num >1:
            train_dataset = Repeat(train_dataset, len(train_dataset)*c.repeat_num)
        else:
            pass
        train_val_dataset = Dataset(c.img_size, c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, is_pairset = c.is_pairset, is_train=True)
    else:
        train_dataset = Dataset(c.img_size, c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, is_pairset = c.is_pairset, is_train=True)
        if c.repeat_num >1:
            train_dataset = Repeat(train_dataset, len(train_dataset)*c.repeat_num)
            train_val_dataset = Dataset(c.img_size, c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, is_pairset = c.is_pairset, is_train=True)
        else:
            pass
    test_aug_dataset = Dataset(c.img_size, c.test_aug_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, c.is_pairset, is_train=False)
    test_dataset = Dataset(c.img_size, c.test_data_path, data_cfg.class_name.test, c.norm_mean, c.norm_std, c.is_pairset, is_train=False)


    #
    if (c.nf_aug_ratio_train>0 or c.repeat_num>1) and c.is_train ==True:
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=c.batch_size, shuffle=True, drop_last=c.drop_last, **kwargs)
    else:
        pass
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=c.drop_last, **kwargs)
    test_aug_loader = torch.utils.data.DataLoader(test_aug_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('train/test_aug/test loader length', len(train_loader.dataset), len(test_aug_loader.dataset), len(test_loader.dataset))
    print('train/test_aug/test loader batches', len(train_loader), len(test_aug_loader), len(test_loader))
    txt_file.write(f'\ntrain/test_aug/test loader length {len(train_loader.dataset)}, {len(test_aug_loader.dataset)}, {len(test_loader.dataset)}')
    txt_file.write(f'\ntrain/test_aug/test loader batches {len(train_loader)}, {len(test_aug_loader)}, {len(test_loader)}')


    #============================================
    #               Set Metrics
    #============================================
     # stats
    if c.is_train==True:
        det_roc_obs = None
        seg_roc_obs = None
        seg_pr_obs = {'exp-layerwise': Score_Observer('PIX_AUPR (exp-layerwise)'),
                    'exp_std-layerwise': Score_Observer('PIX_AUPR (exp_std-layerwise)'),
                    'gauss_cdf-layerwise': Score_Observer('PIX_AUPR (gauss_cdf-layerwise)')}
    else: 
        if c.test_data_type=='aug':     
            aug_det_roc_obs = Score_Observer('DET_AUROC (AUG)')
            aug_seg_roc_obs = Score_Observer('PIX_AUROC (AUG)')
            aug_seg_pr_obs = Score_Observer('PIX_AUPR (AUG)')
        else:
            det_roc_obs = Score_Observer('DET_AUROC')
            seg_roc_obs = Score_Observer('PIX_AUROC')
            seg_pr_obs = Score_Observer('PIX_AUPR')

    # set loss function for enc-dec network
    if c.loss_type == 'cls':
        if c.w_defect>0:
            class_weights = torch.FloatTensor([1-c.w_defect, c.w_defect]).to(c.device)
        else:
            class_weights = torch.FloatTensor([0.5, 0.5]).to(c.device)
        fe_loss_fn = nn.CrossEntropyLoss(weight = class_weights, reduction='none') 
    elif c.loss_type == 'reg': 
        fe_loss_fn = nn.L1Loss(reduction='none') 
    elif 'smooth' in c.loss_type:
        fe_loss_fn = nn.BCELoss(reduction='none') 
    else:
        raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))

    #============================================
    #               Train or Test
    #============================================
    result_dict = dict()
    for epoch in range(c.meta_epochs):
        if c.is_train ==True:
            print('Train meta epoch: {}'.format(epoch))
            txt_file.write('\n\nTrain meta epoch: {}'.format(epoch))
            train_meta_epoch(c, model_cfg, epoch, train_loader, model, optimizers, pool_layers_names, txt_file)
            if epoch % c.eval_epoch == c.eval_epoch-1 or epoch == c.meta_epochs+c.freeze_enc_epochs-1:
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, feature_map_list, pred_list= test_meta_epoch(c, model_cfg, epoch, test_loader, model, pool_layers_names, fe_loss_fn, txt_file)
                if c.nf_aug_ratio_train>0 or c.repeat_num>1:
                    _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, model_cfg, epoch, train_val_loader, model,  pool_layers_names, fe_loss_fn, txt_file)
                else:
                    _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, model_cfg, epoch, train_loader, model, pool_layers_names, fe_loss_fn, txt_file)
        else:
            if 'joint' in c.infer_type:
                cal_model_size_MB(model, result_dict)
            else:
                cal_model_size_MB([encoder, nfs], result_dict)
            state = load_weights(c, model, c.infer_type)
            c_loaded = state['args']
            epoch = c_loaded.meta_epochs 
            if c.test_data_type == 'aug':
                if c.th_manual>0 or c.fn >=0:
                    pass
                else:
                    test_meta_fps(c, model_cfg, epoch, test_aug_loader, model, pool_layers_names, fe_loss_fn, txt_file, result_dict)
                _, _, _, train_dist, _, train_gt_mask_list, train_feature_map_list, train_pred_list= test_meta_epoch(c, model_cfg, epoch, train_loader, model, pool_layers_names, fe_loss_fn, txt_file)
                height, width, test_aug_image_list, test_aug_dist, gt_aug_label_list, gt_aug_mask_list, feature_map_aug_list, pred_aug_list= test_meta_epoch(c, model_cfg, epoch, test_aug_loader, model, pool_layers_names, fe_loss_fn, txt_file)
            else:
                if c.th_manual>0 or c.fn >=0:
                    pass
                else:
                    test_meta_fps(c, model_cfg, epoch, test_loader, model, pool_layers_names, fe_loss_fn, txt_file, result_dict)
                _, _, _, train_dist, _, train_gt_mask_list, train_feature_map_list, train_pred_list= test_meta_epoch(c, model_cfg, epoch, train_loader, model, pool_layers_names, fe_loss_fn, txt_file)
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, feature_map_list, pred_list= test_meta_epoch(c, model_cfg, epoch, test_loader, model, pool_layers_names, fe_loss_fn, txt_file)

        # get test results and export visulaizations
        if c.is_train ==True:
            if epoch % c.eval_epoch == c.eval_epoch-1 or epoch == c.meta_epochs+c.freeze_enc_epochs-1:
                anomaly_cal = Anomaly_Score_Calculator(c.nf_layers, c.crp_size, height, width, [], train_dist, train_gt_mask_list, c.train_type, c.pro) 
                anomaly_cal.save_results(c, epoch, [], test_dist, [], gt_mask_list, gt_label_list, [], test_dataset, data_cfg.full_dims.train, det_roc_obs, seg_roc_obs, seg_pr_obs['exp-layerwise'], '', txt_file, result_dict, indiv_pr_obs=indiv_pr_obs['exp-layerwise'], prob_mapping='exp', inverting_mode='layerwise')
                # anomaly_cal.save_results(c, epoch, [], test_dist, [], gt_mask_list, gt_label_list, [], test_dataset, data_cfg.full_dims.train, det_roc_obs, seg_roc_obs, seg_pr_obs['exp_std-layerwise'], '', txt_file, result_dict, indiv_pr_obs=indiv_pr_obs['exp_std-layerwise'], prob_mapping='exp_std', inverting_mode='layerwise')
                # anomaly_cal.save_results(c, epoch, [], test_dist, [], gt_mask_list, gt_label_list, [], test_dataset, data_cfg.full_dims.train, det_roc_obs, seg_roc_obs, seg_pr_obs['gauss_cdf-layerwise'], '', txt_file, result_dict, indiv_pr_obs=indiv_pr_obs['gauss_cdf-layerwise'], prob_mapping='gauss_cdf', inverting_mode='layerwise')
            save_weights(c, model, c.model_dir) 
        else:
            anomaly_cal = Anomaly_Score_Calculator(c.nf_layers, c.crp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, c.infer_type, c.pro, c.w_fe, c.get_best_w_fe) 
            if c.test_data_type == 'aug':
                if c.th_manual>0 or c.fn >=0 or c.save_features ==True:
                    anomaly_cal.save_results(c, epoch, test_aug_image_list, test_aug_dist, pred_aug_list, gt_aug_mask_list, gt_aug_label_list, feature_map_aug_list, test_aug_dataset, data_cfg.full_dims.train, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, False, indiv_pr_obs=[], prob_mapping='exp', inverting_mode='layerwise')
                else:
                    anomaly_cal.save_results(c, epoch, test_aug_image_list, test_aug_dist, pred_aug_list, gt_aug_mask_list, gt_aug_label_list, feature_map_aug_list, test_aug_dataset, data_cfg.full_dims.train, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, True, indiv_pr_obs=[], prob_mapping='exp', inverting_mode='layerwise')
                if c.fn>=0:
                    threshold = get_threshold_with_fixed_fn(c, anomaly_cal.super_mask_full, anomaly_cal.gt_mask_bool_full, save_dir, data_cfg.class_name.train) 
                    c.th_manual = threshold
                if c.viz:
                    viz(c, data_cfg, test_aug_loader, anomaly_cal, txt_file, save_dir, result_dict)
            else:
                if c.th_manual>0 or c.fn >=0 or c.save_features ==True:
                    anomaly_cal.save_results(c, epoch, test_image_list, test_dist, pred_list, gt_mask_list, gt_label_list, feature_map_list, test_dataset, data_cfg.full_dims.test, det_roc_obs, seg_roc_obs, seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, False, indiv_pr_obs=indiv_pr_obs, prob_mapping='exp', inverting_mode='layerwise')
                else:
                    anomaly_cal.save_results(c, epoch, test_image_list, test_dist, pred_list, gt_mask_list, gt_label_list, feature_map_list, test_dataset, data_cfg.full_dims.test, det_roc_obs, seg_roc_obs, seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, True, indiv_pr_obs=indiv_pr_obs, prob_mapping='exp', inverting_mode='layerwise')
                if c.fn>=0:
                    threshold = get_threshold_with_fixed_fn(c, anomaly_cal.super_mask_full, anomaly_cal.gt_mask_bool_full, save_dir, data_cfg.class_name.test) 
                    c.th_manual = threshold
                if c.viz:
                    viz(c, data_cfg, test_loader, anomaly_cal, txt_file, save_dir, result_dict)
            if c.viz:
                plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, 'normal')
            break
    return log_txt_path
