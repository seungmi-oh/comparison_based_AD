import os, time, gc, copy, random
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from .evaluate import *
from utils import *
from custom_datasets import *
from networks import *


# train an encoder-decoder network to finetune a feature extractor of CNF networks 
def train_meta_epoch(c, model_cfg, epoch, loader, model, optimizers, pool_layers_names, fe_loss_fn, txt_file):
    I = len(loader)
    iterator = iter(loader)

    encoder = model[0]
    decoder = model[1]
    if c.freeze_bn ==True:
        encoder.freeze_bn_layers()
    else:
        pass

    if epoch < c.freeze_enc_epochs:
        encoder = encoder.eval()
    else:
        if c.eval_bn ==True:
            encoder = encoder.eval()
        else:
            encoder = encoder.train()

    # update learning rate
    adjust_learning_rate(c, optimizers, epoch, c.meta_epochs+c.freeze_enc_epochs)
    decoder = decoder.train()

    def get_data_inf():
         while True:
             for out in enumerate(loader):
                 yield out
    dataloader_inf =  get_data_inf()

    for sub_epoch in range(c.sub_epochs):
        train_total_loss = 0.0
        train_defect_losses = [0.0 for i in range(1, c.num_class)]
        train_good_loss = 0.0
        train_total_count = 0
        train_defect_counts = [0 for i in range(1, c.num_class)]
        train_good_count = 0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizers)

            batch_idx, (image, label, mask) = next(dataloader_inf)

            image = image.to(c.device)  # single scale
            label = label.to(c.device)
            if c.loss_type =='cls':
                mask = (mask>0).type(torch.long)
            else:
                pass
            mask = mask.to(c.device)
            
            # encoder
            if epoch < c.freeze_enc_epochs:
                # train decoder only
                with torch.no_grad():
                    enc_out = encoder(image)
            else:
                enc_out = encoder(image)

            # decoder
            if model_cfg.backbone.skip_connection ==True:
                pix_out = decoder(enc_out)
            else:
                pix_out = decoder(enc_out[-1])

            # calculate loss 
            if c.loss_type=='cls':
                pix_loss = fe_loss_fn(pix_out, torch.squeeze(mask,1))
                pix_loss = torch.unsqueeze(pix_loss,1)
            elif c.loss_type =='reg':
                pix_loss = torch.sum(fe_loss_fn(pix_out, image), 1, keepdim=True)
            elif 'smooth' in c.loss_type:
                pix_loss = fe_loss_fn(pix_out, mask)
            else:
                raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))

            train_defect_losses[0] += torch.sum(pix_loss[mask!=0]).item()  
            train_defect_counts[0] += torch.sum(mask!=0).item() 
            train_good_loss += torch.sum(pix_loss[mask==0]).item() 
            train_good_count += torch.sum(mask==0).item() 
            batch_loss = pix_loss.mean()

            optimizers.zero_grad()
            batch_loss.backward()
            optimizers.step()

            train_total_loss += t2np(batch_loss*len(label)) 
            train_total_count += len(label)

        # summary training progress
        mean_train_total_loss = train_total_loss / train_total_count
        mean_train_good_loss = train_good_loss / train_good_count
        defect_str = ''
        for cl in range(len(train_defect_losses)):
            mean_train_defect_loss = train_defect_losses[cl] / train_defect_counts[cl]
            defect_str += f'train_defect{cl+1}_loss: {mean_train_defect_loss:.4f}, '

        if c.verbose:
            print(f'Epoch: {epoch:02d}.{sub_epoch:03d}\ttrain_total_loss: {mean_train_total_loss:.4f}, {defect_str}train_good_loss: {mean_train_good_loss:.4f}, lr={lr:.6f}')
        txt_file.write(f'\nEpoch: {epoch:02d}.{sub_epoch:03d}\ttrain_total_loss: {mean_train_total_loss:.4f}, {defect_str}train_good_loss: {mean_train_good_loss:.4f}, lr={lr:.6f}')


def test_meta_epoch(c, model_cfg, epoch, loader, model, pool_layers_names, fe_loss_fn, txt_file):
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    txt_file.write('\nCompute loss and scores on test set:')
    #
    encoder = model[0]
    decoder = model[1]
    encoder = encoder.eval()
    decoder = decoder.eval()
    feature_maps = []

    image_list = list()
    if 'cls' in c.loss_type:
        pred_list = list()
    elif c.loss_type =='reg':
        pred_list = [list(), list()]
    else:
        raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    gt_label_list = list()
    gt_mask_list = list()

    test_total_loss = 0.0
    test_defect_losses = [0.0]
    test_good_loss = 0.0
    test_total_count = 0
    test_defect_counts = [0.0]
    test_good_count = 0

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

            # save feature map
            if c.is_train ==False and c.viz_total_features ==True:
                f_num=0
                for l, layer in enumerate(pool_layers_names):
                    e = dec_activation[layer].detach()  # bxcxhxw
                    
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

            # calculate loss for enc-dec
            if c.loss_type =='cls':
                pix_loss = fe_loss_fn(pix_out, torch.squeeze(mask,1))
                pix_loss = torch.unsqueeze(pix_loss,1)
                prob_map = torch.softmax(pix_out, 1)
                pred_list.extend(t2np(1-prob_map[:,0,:,:]))
            elif c.loss_type =='reg':
                pix_loss = torch.sum(fe_loss_fn(pix_out, image), 1, keepdim=True)
                pred_list[0].extend(t2np(pix_out))
                pred_list[1].extend(t2np(torch.squeeze(pix_loss,1)))
            elif 'smooth' in c.loss_type:
                pix_loss = fe_loss_fn(pix_out, mask.type(torch.float))
                pred_list.extend(t2np(torch.squeeze(pix_out,1)))
            else:
                raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
            for cl in range(len(test_defect_losses)):
                test_defect_losses[cl] += torch.sum(pix_loss[mask>0]).item()  
                test_defect_counts[cl] += torch.sum(mask>0).item() 
            test_good_loss += torch.sum(pix_loss[mask==0]).item() 
            test_good_count += torch.sum(mask==0).item() 
            batch_loss = pix_loss.mean()

            test_total_loss += t2np(batch_loss*len(label)) 
            test_total_count += len(label)

    # show results
    mean_test_total_loss = test_total_loss / test_total_count
    mean_test_good_loss = test_good_loss / test_good_count
    defect_str = ''
    for cl in range(len(test_defect_losses)):
        mean_test_defect_loss = test_defect_losses[cl] / test_defect_counts[cl]
        defect_str += f'test_defect{cl+1}_loss: {mean_test_defect_loss:.4f}, '

    if c.verbose:
        print(f'Epoch: {epoch:02d}\ttest_total_loss: {mean_test_total_loss:.4f}, {defect_str}test_good_loss: {mean_test_good_loss:.4f}')
    txt_file.write(f'\nEpoch: {epoch:02d}\ttest_total_loss: {mean_test_total_loss:.4f}, {defect_str}test_good_loss: {mean_test_good_loss:.4f}')
    return image_list, gt_label_list, gt_mask_list, pred_list, feature_maps


def test_meta_fps(c, model_cfg, epoch, loader, model, pool_layers_names, fe_loss_fn, txt_file, result_dict):
    if c.verbose:
        print('\nCompute inference speed (fps) on test set:')
    txt_file.write('\nCompute inference speed (fps) on test set:')
    #
    encoder = model[0]
    decoder = model[1]
    encoder = encoder.eval()
    decoder = decoder.eval()

    starter, ender = time_measure()
    with torch.no_grad():
        # warm-up
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            image = image.to(c.device) 
            _ = encoder(image)  
            if c.add_fe_anomaly==True:
                if model_cfg.backbone.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])
            else:
                pass

        # measure inference time
        starter.record()
        for k in range(c.repeat_num):
            for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                image = image.to(c.device) # single scale
                enc_out = encoder(image)
                if model_cfg.backbone.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])

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

        map_type = c.loss_type 

        if len(dir_feature)>0:
            str_dir_features = '-'.join(dir_feature)
            tag = os.path.join(map_type, str_dir_features)
        else:
            tag = os.path.join(map_type, 'classic')
        save_dir = os.path.join(c.model_dir, c.data_settings, data_cfg.class_name.test, c.infer_type, tag)
        makedirs(save_dir)
        log_txt_path = os.path.join(save_dir, f'{c.infer_type}_test_results.txt')
    txt_file = open(log_txt_path, 'a')


    #============================================
    #               Set Model
    #============================================

    # set encoder
    encoder, pool_layers_names, pool_dims_total = load_encoder_arch(c, model_cfg.backbone.skip_layers, model_cfg)
    encoder = encoder.eval()
    if c.is_train == True:
        sample = torch.zeros(tuple([2]+c.img_dims))
        _ = encoder(sample)
        model_file.write('Encoder Summary \n')
        model_file.write(str(encoder))
    else:
        pass
    encoder = encoder.to(c.device)

    # set decoder
    if c.is_train == True:
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
    if c.is_train == True:
        model_file.write('Decoder Summary \n')
        if model_cfg.backbone.skip_connection ==True:
            model_file.write(str(decoder))
        else:
            dec_stat=summary(decoder,c.dec_dims_fe[-1], depth=5)
            model_file.write(str(dec_stat))
    else:
        pass

    decoder = decoder.to(c.device)

    model = [encoder, decoder] 

    optimizers = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=c.lr, weight_decay=c.w_decay)

    if c.is_train == True:
        model_file.close()

    #============================================
    #               Set Data
    #============================================
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    # task data
    if c.aug_ratio_train>0:
        train_dataset = PerlinDefectsTrainDataset(c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, c.aug_ratio_train, c.use_in_domain_data, c.img_size, c.anomaly_size, c.anomaly_confidence, c.is_pairset, c.loss_type) 
        if c.repeat_num >1:
            train_dataset = Repeat(train_dataset, len(train_dataset)*c.repeat_num)
        else:
            pass
    else:
        train_dataset = Dataset(c.img_size, c.train_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, is_pairset = c.is_pairset, is_train=True)
        if c.repeat_num >1:
            train_dataset = Repeat(train_dataset, len(train_dataset)*c.repeat_num)
        else:
            pass

    test_aug_dataset = Dataset(c.img_size, c.test_aug_data_path, data_cfg.class_name.train, c.norm_mean, c.norm_std, c.is_pairset, is_train=False)
    test_dataset = Dataset(c.img_size, c.test_data_path, data_cfg.class_name.test, c.norm_mean, c.norm_std, c.is_pairset, is_train=False, init_chip = c.init_chip, fin_chip = c.fin_chip)

    #
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
    if c.is_train ==True:
        det_roc_obs = None
        seg_roc_obs = None
        seg_pr_obs = Score_Observer('SEG_AUPR')
        aug_det_roc_obs = None
        aug_seg_roc_obs = None
        aug_seg_pr_obs = Score_Observer('SEG_AUPR (AUG)')
    else:
        det_roc_obs = Score_Observer('DET_AUROC')
        seg_roc_obs = Score_Observer('SEG_AUROC')
        seg_pr_obs = Score_Observer('SEG_AUPR')
        aug_det_roc_obs = Score_Observer('DET_AUROC (AUG)')
        aug_seg_roc_obs = Score_Observer('SEG_AUROC (AUG)')
        aug_seg_pr_obs = Score_Observer('SEG_AUPR (AUG)')
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
    for epoch in range(c.meta_epochs+c.freeze_enc_epochs):
        if c.is_train==True:
            print('Train meta epoch: {}'.format(epoch))
            txt_file.write('\n\nTrain meta epoch: {}'.format(epoch))
            train_meta_epoch(c, model_cfg, epoch, train_loader, model, optimizers, c.pool_layers_dec, fe_loss_fn, txt_file)
            if epoch % c.eval_epoch == c.eval_epoch-1 or epoch == c.meta_epochs+c.freeze_enc_epochs-1:
                test_aug_image_list, gt_aug_label_list, gt_aug_mask_list, pred_aug_list, feature_map_aug_list= test_meta_epoch(c, model_cfg, epoch, test_aug_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file)
                test_image_list, gt_label_list, gt_mask_list, pred_list, feature_map_list= test_meta_epoch(c, model_cfg, epoch, test_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file)
        else:
            cal_model_size_MB(model, result_dict)
            state = load_weights(c, model, c.infer_type)
            c_loaded = state['args']
            epoch = c_loaded.meta_epochs 
            if c.test_data_type == 'aug':
                if c.th_manual>0 or c.fn >=0:
                    pass
                else:
                    test_meta_fps(c, model_cfg, epoch, test_aug_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file, result_dict)
                test_aug_image_list, gt_aug_label_list, gt_aug_mask_list, pred_aug_list, feature_map_aug_list= test_meta_epoch(c, model_cfg, epoch, test_aug_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file)
            else:
                if c.th_manual>0 or c.fn >=0:
                    pass
                else:
                    test_meta_fps(c, model_cfg, epoch, test_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file, result_dict)
                test_image_list, gt_label_list, gt_mask_list, pred_list, feature_map_list= test_meta_epoch(c, model_cfg, epoch, test_loader, model, c.pool_layers_dec, fe_loss_fn, txt_file)

        # get test results and export visulaizations
        if c.is_train==True:
            if epoch % c.eval_epoch == c.eval_epoch-1 or epoch == c.meta_epochs+c.freeze_enc_epochs-1:
                anomaly_cal = Anomaly_Score_Calculator([], c.crp_size, [], [], [], [], [], c.train_type, c.pro) 
                anomaly_cal.save_results(c, epoch, [], [], pred_aug_list, gt_aug_mask_list, gt_aug_label_list, [], test_aug_dataset, data_cfg.full_dims.train, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, '', txt_file, result_dict)
                anomaly_cal.save_results(c, epoch, [], [], pred_list, gt_mask_list, gt_label_list, [], test_dataset, data_cfg.full_dims.train, det_roc_obs, seg_roc_obs, seg_pr_obs, '', txt_file, result_dict)
            save_weights(c, model, c.model_dir)  
        else:
            anomaly_cal = Anomaly_Score_Calculator([], c.crp_size, [], [], [],[], [], c.infer_type, c.pro) 
            if c.test_data_type == 'aug':
                if c.th_manual>0 or c.fn >=0:
                    anomaly_cal.save_results(c, epoch, test_aug_image_list, [], pred_aug_list, gt_aug_mask_list, gt_aug_label_list, feature_map_aug_list, test_aug_dataset, data_cfg.full_dims.train, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, False)
                else:
                    anomaly_cal.save_results(c, epoch, test_aug_image_list, [], pred_aug_list, gt_aug_mask_list, gt_aug_label_list, feature_map_aug_list, test_aug_dataset, data_cfg.full_dims.train, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, True)
                if c.fn>=0:
                    threshold = get_threshold_with_fixed_fn(c, anomaly_cal.super_mask_full, anomaly_cal.gt_mask_bool_full, save_dir, data_cfg.class_name.train) 
                    c.th_manual = threshold
                if c.viz:
                    viz(c, data_cfg, test_aug_loader, anomaly_cal, txt_file, save_dir, result_dict)
            else:
                if c.th_manual>0 or c.fn >=0:
                    anomaly_cal.save_results(c, epoch, test_image_list, [], pred_list, gt_mask_list, gt_label_list, feature_map_list, test_dataset, data_cfg.full_dims.test, det_roc_obs, seg_roc_obs, seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, False)
                else:
                    anomaly_cal.save_results(c, epoch, test_image_list, [], pred_list, gt_mask_list, gt_label_list, feature_map_list, test_dataset, data_cfg.full_dims.test, det_roc_obs, seg_roc_obs, seg_pr_obs, save_dir, txt_file, result_dict, c.is_full, True)
                if c.fn>=0:
                    threshold = get_threshold_with_fixed_fn(c, anomaly_cal.super_mask_full, anomaly_cal.gt_mask_bool_full, save_dir, data_cfg.class_name.test) 
                    c.th_manual = threshold
                if c.viz:
                    viz(c, data_cfg, test_loader, anomaly_cal, txt_file, save_dir, result_dict)
            break

    return log_txt_path

            
