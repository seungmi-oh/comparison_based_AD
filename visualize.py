'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os
import random
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve, f1_score, auc
from skimage import morphology
from skimage.segmentation import mark_boundaries
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from utils import *

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


# plot histogram
def export_hist(c, gts, scores, detect_type, save_dir):
    image_dirs = os.path.join(save_dir, 'stats')
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)

    Y = scores.flatten()
    Y_label = gts.flatten()

    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    defect_num = np.sum(Y_label == 1)

    plt.hist(
    [Y[Y_label == 1], Y[Y_label == 0]],
    bins=500,
    range=(0, 1),           # 0부터 1까지 bin 고정
    align='left',           # 막대 위치 왼쪽 정렬
    density=True,
    color=['r', 'g'],
    label=['defective', 'normal'],
    alpha=0.3,
    histtype='stepfilled')

    plt.legend(frameon=False)  # ← legend 테두리 제거
    plt.xlim(-0.1, 1.1)             # ← x축 범위 제한

    # 로그 스케일 y축
    log_base = 2
    ax.set_yscale('symlog', linthresh=1, base=2, linscale=0.5)
    # 현재 y범위에서 2^n 리스트 계산
    
    lo, hi = ax.get_ylim()
    lo = max(lo, 1)  # 1 이하는 라벨 숨길 거라 1부터
    exps = np.arange(
        np.ceil(np.log(lo)/np.log(log_base)),
        np.floor(np.log(hi)/np.log(log_base)) + 1
    )
    ticks = (log_base ** exps).astype(float)
    ax.set_yticks(ticks)

    def hide_ticks_below_one(x, pos):
        return "" if x < 1 else f"{int(x)}"

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(hide_ticks_below_one))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    image_file = os.path.join(
        image_dirs, f'hist_plots_{detect_type}_{c.run_date.replace(":","")}.pdf'
    )
    fig.savefig(image_file, dpi=dpi, format='pdf', 
                bbox_inches='tight', pad_inches=0.0)
    plt.close()


# save visualization results
def export_test_images(c, data_cfg, test_img, gts, scores, pix_imgs, feat_maps, threshold, file_list, image_dirs, txt_file, w_fe, result_dict=None):
    print('Exporting images...')
    num = len(file_list)
    if c.is_k_disk==True:
        kernel = morphology.disk(c.k_size)
    else:
        kernel = np.ones(c.k_size)
    if 'fe_only' in c.infer_type:
        scores_norm = 1.0
    else:
        scores_norm = 1.0/scores.max()
    seg_performance = [0,0,0] 
    jet_8 = plt.get_cmap('jet', lut=2**8)

    feat_normalization_params = []
    for feat_map in feat_maps:
        feat_map = np.array(feat_map)
        feat_max = feat_map.max()
        feat_min = feat_map.min()
        feat_normalization_params.append((feat_min, feat_max))

    for i in range(num):
        print(f'{i+1}/{num}')
        # image
        if c.is_pairset ==True: 
            img1 = test_img[0][i]
            img1 = denormalization(img1, c.norm_mean, c.norm_std)
            img2 = test_img[1][i]
            img2 = denormalization(img2, c.norm_mean, c.norm_std)
            img_list = [img1, img2]
        else:
            if len(test_img)==1:
                img = test_img[0][i]
            else:
                img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            img_list = [img]

        if len(pix_imgs)==0:
            preds_img = []
        elif len(pix_imgs)==num:
            preds = pix_imgs[i]
            pred_img = (255.0*preds).astype(np.uint8)
            preds_img_8 = jet_8(pred_img)
            preds_img = [np.uint8(preds_img_8[:,:,:3]*255)]
        elif len(pix_imgs)==1:
            preds = pix_imgs[0][i]
            pred_img = (255.0*preds).astype(np.uint8)
            preds_img_8 = jet_8(pred_img)
            preds_img = [np.uint8(preds_img_8[:,:,:3]*255)]
        elif len(pix_imgs)==2:
            preds = pix_imgs[1][i]
            pred_img = (255.0*preds).astype(np.uint8)
            preds_img_8 = jet_8(pred_img)
            preds_img_ = np.uint8(preds_img_8[:,:,:3]*255)
            recons = denormalization(pix_imgs[0][i], c.norm_mean, c.norm_std)
            recons_bgr = cv2.cvtColor(recons, cv2.COLOR_RGB2BGR)
            preds_img = [recons_bgr, preds_img_]
        else: 
            print(len(pix_imgs))
            raise KeyboardInterrupt

       # gts
        gt_mask = gts[i].astype(np.float32)
        if len(gt_mask.shape)==3:
            gt_mask = np.transpose(gt_mask, (1,2,0))
        else:
            pass
        gt_mask = (255.0*gt_mask).astype(np.uint8)
        dilate_kernel = np.ones((3,3), dtype=np.uint8)
        gt_mask = cv2.dilate(gt_mask, dilate_kernel, iterations=1)

        # anomaly prediction
        score_mask = np.zeros_like(scores[i])
        score_mask[scores[i] >=  threshold] = 1.0
        score_mask = (255.0*score_mask).astype(np.uint8)
        if c.is_close == True and c.is_k_disk==True:
            score_mask = morphology.closing(score_mask, kernel)
        if c.is_open == True and c.is_k_disk ==True:
            score_mask = morphology.opening(score_mask, kernel)
        score_mask = (255.0*score_mask).astype(np.uint8)

        # scores
        if len(preds_img)>0 and ('fe_only' in c.infer_type) == False:
            # aggregated scores
            if c.add_fe_anomaly==True:
                score_nf = scores[i]-w_fe*preds
                score_nf_norm = 1.0/score_nf.max()
                score_map = (255.0*score_nf*score_nf_norm).astype(np.uint8)
                score_map_8 = jet_8(score_map)
                score_map = score_map_8[:,:,:3]*255

                scores_tot_norm = 1.0/scores.max()
                score_map_tot = (255.0*scores[i]*scores_tot_norm).astype(np.uint8)
                score_map_tot_8 = jet_8(score_map_tot)
                score_map_tot = score_map_tot_8[:,:,:3]*255
                score_map_list = [score_map, score_map_tot]
            # nf score only
            else:
                score_nf = scores[i]
                score_nf_norm = 1.0/score_nf.max()
                score_map = (255.0*score_nf*score_nf_norm).astype(np.uint8)
                score_map_8 = jet_8(score_map)
                score_map = score_map_8[:,:,:3]*255
                score_map_list = [score_map]
        # fe score
        else:
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            score_map_8 = jet_8(score_map)
            score_map = score_map_8[:,:,:3]*255
            score_map_list = [score_map]

        # feature maps
        if len(feat_maps) != len(scores):
            normed_feat_maps = []
            l=0
            for feat_map_ in feat_maps: 
                # mean_feat_map = feat_map_[i] / feat_normalization_params[l][1]
                # mean_normed_feat_map= (255*rescale(mean_feat_map)).astype(np.uint8)
                mean_feat_map = feat_map_[i]
                mean_normed_feat_map = (255*(mean_feat_map-feat_normalization_params[l][0])/(feat_normalization_params[l][1]-feat_normalization_params[l][0])).astype(np.uint8)
                mean_feat_map_8 = jet_8(mean_normed_feat_map)
                mean_normed_feat_map = mean_feat_map_8[:,:,:3]*255
                normed_feat_maps.append(mean_normed_feat_map)
                l=l+1 
        else:
            feat_map = feat_maps[i]
            if len(feat_map.shape)>2:
                feat_map = np.mean(feat_maps[i],0)
            else:
                pass
            normed_feat_map = (255*rescale(feat_map)).astype(np.uint8)
            feat_map_8 = jet_8(normed_feat_map)
            normed_feat_map = feat_map_8[:,:,:3]*255
            normed_feat_maps = [normed_feat_map]

        # get seg-wise results
#        if c.is_full==True:
#            save_file_name = file_list[i]
#            if c.save_all_tn_sample ==True:
#                seg_img_list, seg_pred_list, seg_score_list, seg_smap_list, seg_gt_list, seg_feat_list, seg_file_list, img_seg_performance = get_segment_img(img_list, preds_img, score_mask, score_map_list, gt_mask, normed_feat_maps, save_file_name, image_dirs, c.img_dims, c.th_pix, True, True) 
#            else:
#                if i<5:
#                    seg_img_list, seg_pred_list, seg_score_list, seg_smap_list, seg_gt_list, seg_feat_list, seg_file_list, img_seg_performance = get_segment_img(img_list, preds_img, score_mask, score_map_list, gt_mask, normed_feat_maps, save_file_name, image_dirs, c.img_dims, c.th_pix, True, True) 
#                else:
#                    seg_img_list, seg_pred_list, seg_score_list, seg_smap_list, seg_gt_list, seg_feat_list, seg_file_list, img_seg_performance = get_segment_img(img_list, preds_img, score_mask, score_map_list, gt_mask, normed_feat_maps, save_file_name, image_dirs, c.img_dims, c.th_pix, True) 
#        else:
        image_file_path = file_list[i]
        save_file_info_ = image_file_path.replace(os.path.join(os.path.join(c.data_path, data_cfg.class_name.test), 'test')+os.path.sep,'') 
        save_file_info = save_file_info_[:-4]
        save_file_info_list = save_file_info.split(os.path.sep) 
        save_file_name = '-'.join(save_file_info_list[-2:])
        seg_img_list, seg_pred_list, seg_score_list, seg_smap_list, seg_gt_list, seg_feat_list, seg_file_list, img_seg_performance = get_segment_img(img_list, preds_img, score_mask, score_map_list, gt_mask, normed_feat_maps, save_file_name, image_dirs, c.img_dims, c.th_pix, False) 


        for e in range(len(seg_performance)):
            seg_performance[e] += img_seg_performance[e]

        # detection result 
        for seg_i in range(len(seg_img_list)):
            c_img = seg_img_list[seg_i] 
            c_pred = seg_pred_list[seg_i]
            c_gt_mask = seg_gt_list[seg_i] 
            c_score_mask = seg_score_list[seg_i]*255 
            c_score_map = seg_smap_list[seg_i] 
            c_feat_map = seg_feat_list[seg_i]
        
            c_score_mask_3D = np.transpose(np.array([c_score_mask]*3, dtype=np.uint8), (1,2,0))
            c_score_img_bgr = np.array(c_score_mask_3D, np.uint8)

            if c_pred is None:
                save_img = np.concatenate((c_img, c_gt_mask, c_score_map, c_score_img_bgr), axis=1)
            else:
                save_img = np.concatenate((c_img, c_gt_mask, c_pred,  c_score_map, c_score_img_bgr), axis=1)
            
            final_img1 = []
            final_img2 = []
            interval = np.ones((save_img.shape[0], 10, 3))*255
            for s in range(int(save_img.shape[1]/c_gt_mask.shape[1])):
                final_img1.append(save_img[:, s*c_gt_mask.shape[1]:(s+1)*c_gt_mask.shape[1], :])
                final_img1.append(interval)
                
            if c_feat_map is None:
                final_img = np.concatenate(final_img1[:-1], axis=1)
            else:
                for f in range(int(c_feat_map.shape[1]/c_gt_mask.shape[1])):
                    final_img2.append(c_feat_map[:, f*c_gt_mask.shape[1]:(f+1)*c_gt_mask.shape[1], :])
                    final_img2.append(interval)

                if len(c.att_type)==0:
                    feat_num = int((c_feat_map.shape[1]/c_gt_mask.shape[1])/len(c.nf_layers))
                    if feat_num != 1:
                        final_img1 = np.concatenate(final_img1[:-1], axis=1)
                        for l in range(len(c.nf_layers)):
                            feat_img = np.concatenate(final_img2[l*feat_num*2:(l+1)*feat_num*2-1], axis=1)
                            v_interval_width = max(feat_img.shape[1], final_img1.shape[1])
                            v_interval = np.ones((10, v_interval_width, 3))*255
                            if v_interval_width == feat_img.shape[1]:
                                if l==0:
                                    final_img1_ = np.ones_like(feat_img)*255
                                    final_img1_[:, :final_img1.shape[1], :] = final_img1
                                else:
                                    pass
                                feat_img_ = feat_img
                            else:
                                if l==0:
                                    final_img1_ = final_img1
                                else:
                                    pass
                                feat_img_ = np.ones_like(final_img1)*255
                                feat_img_[:, :feat_img.shape[1], :] = feat_img 
                            final_img1_ = np.concatenate([final_img1_, v_interval, feat_img_], axis = 0)
                        final_img = final_img1_
                    else:
                        final_img = final_img1+final_img2
                        final_img = np.concatenate(final_img[:-1], axis=1)
                # visulaize with features of attention modules
                else:
                    normal_layers = list(set(c.nf_layers)-set(c.att_layers)) 
                    total_feat_num = int(c_feat_map.shape[1]/c_gt_mask.shape[1])
                    if c.viz_total_features==True:
                        normal_feat_num = 3 
                    if c.viz_diff_features==True:
                        normal_feat_num = 1 
                    if (c.viz_diff_features ==False) and (c.viz_total_features==False):
                        raise NotImplementedError('You should set viz_diff_features or viz_total_features to True.')
                    att_feat_num = int((total_feat_num-len(normal_layers)*normal_feat_num)/len(c.att_layers))
                    final_img1 = np.concatenate(final_img1[:-1], axis=1)
                    f_init = 0 
                    f_fin = 0 
                    for l in range(len(c.nf_layers)):
                        if c.nf_layers[l] in c.att_layers:
                            f_last = f_init+att_feat_num
                        else:
                            f_last = f_init+normal_feat_num
                        feat_img = np.concatenate(final_img2[f_init*2:f_last*2-1], axis=1)
                        f_init = f_last
                        v_interval_width = max(att_feat_num*c_gt_mask.shape[1]+(att_feat_num-1)*10, final_img1.shape[1])
                        v_interval = np.ones((10, v_interval_width, 3))*255
                        if l==0:
                            if final_img1.shape[1] < v_interval_width:
                                final_img1_ = np.ones((c_gt_mask.shape[0], v_interval_width, 3))*255
                                final_img1_[:, :final_img1.shape[1], :] = final_img1
                            else:
                                final_img1_ = final_img1
                        else:
                            pass

                        if feat_img.shape[1] < v_interval_width:
                            feat_img_ = np.ones((c_gt_mask.shape[0], v_interval_width, 3))*255
                            feat_img_[:, :feat_img.shape[1], :] = feat_img 
                        else:
                            feat_img_ = feat_img
                        final_img1_ = np.concatenate([final_img1_, v_interval, feat_img_], axis = 0)
                    final_img = final_img1_
            final_img = final_img.astype(np.uint8)
            image_file = seg_file_list[seg_i]+'.jpg'
            cv2.imwrite(image_file, final_img)

    # write segmentation result
    if result_dict !=None:
        tp, fn, fp = seg_performance
        recall = tp/(fn+tp+1e-7)
        precision = tp/(fp+tp+1e-7)
        f1_score = 2*recall*precision/(recall+precision+1e-7)
        txt_file.write(f'\nTP num: {tp}, FN num: {fn}, FP num: {fp}')
        txt_file.write(f'\nSEG Performance) recall: {recall*100.0 :0.2f}, precision: {precision*100.0:0.2f}, f1-score: {f1_score*100.0:0.2f}')
        result_dict['threshold'] = threshold
        result_dict['TP'] = tp 
        result_dict['FN'] = fn
        result_dict['FP'] = fp
        result_dict['recall'] = recall*100.0
        result_dict['precision'] = precision*100.0
        result_dict['f1-score'] = f1_score*100.0
        result_dict['fp_over_P'] = (fp/(tp+fn+1e-8))*100.0
        csv_path = os.path.join(image_dirs, 'test_result.csv') 
        write_csv(result_dict, csv_path, data_cfg.class_name.test)


def get_segment_img(test_imgs, test_preds, score_mask, score_maps_list, gt_mask, feat_map, file_name, image_dirs, img_dims, th_pix, full_img = True, tn_sample=False, overlapped_pix=30):
    H=img_dims[-2]
    W=img_dims[-1]
    seg_tp = 0
    seg_fn = 0
    seg_fp = 0

    # image
    gray_test_imgs = []
    bgr_test_imgs = []
    for test_img in test_imgs:
        if np.max(test_img)>1:
            test_img_norm = test_img
        else:
            test_img_norm = cv2.normalize(test_img, None, 0,255, cv2.NORM_MINMAX)
        gray_test_img = cv2.cvtColor(test_img_norm, cv2.COLOR_RGB2GRAY)
        gray_test_imgs.append(gray_test_img)
        bgr_test_img = cv2.cvtColor(test_img_norm, cv2.COLOR_RGB2BGR)
        bgr_test_imgs.append(bgr_test_img)

    if len(gray_test_imgs)>1:
        gray_test_img_interpolated = np.squeeze(0.5*(gray_test_imgs[0] + gray_test_imgs[1]))
        test_img_gray_3D = np.transpose(np.array([gray_test_img_interpolated]*3, dtype=np.uint8), (1,2,0))
    else:
        test_img_gray_3D = np.transpose(np.array([gray_test_imgs[0]]*3, dtype=np.uint8), (1,2,0))

    # scores
    test_smap_bgr_list = []
    for score_map in score_maps_list:
        test_smap_only = np.array(score_map, dtype = np.uint8)
        test_smap = cv2.addWeighted(test_smap_only, 0.7, test_img_gray_3D, 0.3, 0)
        test_smap_bgr = cv2.cvtColor(test_smap, cv2.COLOR_RGB2BGR)
        test_smap_bgr_list.append(test_smap_bgr)

    # gts
    if np.max(gt_mask)>1:
        gt_norm = gt_mask
    else:
        gt_norm = cv2.normalize(gt_mask, None, 0,255, cv2.NORM_MINMAX)
    if gt_norm.shape[-1]==1 or len(gt_norm.shape)==2:
        gt_bgr = np.transpose(np.array([gt_norm]*3, dtype=np.uint8), (1,2,0))
    else:
        gt_bgr = cv2.cvtColor(gt_norm, cv2.COLOR_RGB2BGR)
    
    # feature maps
    test_fmap_bgr = []
    for i in range(len(feat_map)):
        test_fmap_only = np.array(feat_map[i], dtype = np.uint8)
        test_fmap_bgr.append(cv2.cvtColor(test_fmap_only, cv2.COLOR_RGB2BGR))

    # predictions
    bgr_test_preds = []
    if len(test_preds)==0:
        pass
    else:
        test_preds_only = cv2.cvtColor(test_preds[-1], cv2.COLOR_RGB2BGR)
        test_pred_bgr = cv2.addWeighted(test_preds_only, 0.7, test_img_gray_3D, 0.3, 0)
        if len(test_preds)==2:
            bgr_test_preds = [test_preds[0], test_pred_bgr]
        elif len(test_preds)==1:
            bgr_test_preds = [test_pred_bgr]

    # set directory 
    if len(image_dirs)>2:
        makedirs(os.path.join(image_dirs, 'tp'))
        makedirs(os.path.join(image_dirs, 'fp'))
        makedirs(os.path.join(image_dirs, 'fn'))
        makedirs(os.path.join(image_dirs, 'tn'))

    # get seg results
    cropped_img_list = []
    cropped_preds_list = []
    score_mask_list = []
    score_map_list = []
    gt_mask_list = []
    feat_map_list = []
    file_list = []
    
    if len(gt_mask.shape)==3 and gt_mask.shape[-1]>1:
        gt_bool = np.sum(gt_mask, -1)
        gt_bool = np.uint8(gt_bool>0)
    else:
        pass

    # good
    if np.sum(gt_mask)==0 and full_img ==False:
        # true negative
        if np.sum(score_mask)==0:
            cropped_img=np.concatenate(bgr_test_imgs, axis=1)
            if len(bgr_test_preds)==0:
                cropped_preds = None
            elif len(bgr_test_preds)==1:
                cropped_preds = bgr_test_preds[0]
            else:
                cropped_preds_ = []
                for bgr_test_pred in bgr_test_preds:
                    cropped_preds_.append(bgr_test_pred)
                cropped_preds = np.concatenate(cropped_preds_, axis=1) 
            cropped_gt=gt_bgr
            cropped_score=score_mask
            cropped_smaps = []
            for test_smap_bgr in test_smap_bgr_list:
                cropped_smaps.append(test_smap_bgr)
            cropped_smap=np.concatenate(cropped_smaps, axis = 1)
                
            cropped_fmaps = []
            for i in range(len(test_fmap_bgr)):
                cropped_fmaps.append(test_fmap_bgr[i])
            if len(cropped_fmaps)==0:
                cropped_feat_map = None
            else:
                cropped_feat_map = np.concatenate(cropped_fmaps, axis=1) 

            cropped_img_list.append(cropped_img)
            cropped_preds_list.append(cropped_preds)
            score_mask_list.append(cropped_score)
            score_map_list.append(cropped_smap)
            gt_mask_list.append(cropped_gt)
            feat_map_list.append(cropped_feat_map)
            file_list.append(os.path.join(os.path.join(image_dirs, 'tn'), f'{file_name}')) 
        # false positive
        else:
            detect_seg_num, detect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(score_mask)
            seg_fp =detect_seg_num-1
            if len(image_dirs)>2:
                append_segment_img(bbox_info, seg_fp, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fp')
            else:
                pass
    # defect
    elif full_img==True:
        # false positive
        detect_seg_num, detect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(score_mask)
        for seg_idx in range(1, detect_seg_num):
            if np.sum(detect_seg_wise_label==seg_idx)<=th_pix:
                score_mask[detect_seg_wise_label==seg_idx]=0
            else:
                if np.sum(gt_bool[detect_seg_wise_label==seg_idx])>0:
                    if full_img ==True and np.sum(detect_seg_wise_label==seg_idx)>np.prod(score_mask.shape)*0.5:
                        seg_fp = 1000
                    else:
                        pass
                else: 
                    seg_fp += 1
                    if len(image_dirs)>2:
                        append_segment_img(bbox_info, seg_idx, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fp')
                    else:
                        pass
        if tn_sample==True:
            for init_h in range(0, score_mask.shape[-2], H-overlapped_pix):
                for init_w in range(0, score_mask.shape[-1], W-overlapped_pix):
                    fin_h=init_h + H
                    fin_w=init_w + W
                    if fin_h>score_mask.shape[-2]:
                        fin_h=score_mask.shape[-2]
                        init_h=score_mask.shape[-2]-H
                    if fin_w>score_mask.shape[-1]:
                        fin_w=score_mask.shape[-1]
                        init_w=score_mask.shape[-1]-W
                    cropped_score_mask = score_mask[init_h:fin_h, init_w:fin_w]
                    cropped_gt = gt_bgr[init_h:fin_h, init_w:fin_w, :]
                    if (np.sum(cropped_score_mask)==0) and (np.sum(cropped_gt)==0):
                        offset_h, offset_w = int(H*0.2), int(W*0.2)
                        bbox_info = np.array([[init_w, init_h, W, H, 0]])
                        append_segment_img(bbox_info, 0, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'tn')
                    else:
                        pass


            
        defect_seg_num, defect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(gt_bool)
        # false negative or true positive
        for seg_idx in range(1, defect_seg_num):
            if np.sum(score_mask[defect_seg_wise_label==seg_idx])>0:
                if np.sum(score_mask[defect_seg_wise_label==seg_idx])<=th_pix:
                    score_mask[detect_seg_wise_label==seg_idx]=0
                    seg_fn += 1
                else:
                    seg_tp += 1
                    if seg_tp>0:
                        if len(image_dirs)>2:
                            append_segment_img(bbox_info, seg_idx, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'tp')
                        else:
                            pass
            else: 
                seg_fn += 1
                if seg_fn>0:
                    if len(image_dirs)>2:
                        append_segment_img(bbox_info, seg_idx, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fn')
                    else:
                        pass


    else:
        # false positive
        detect_seg_num, detect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(score_mask)
        for seg_idx in range(1, detect_seg_num):
            if np.sum(detect_seg_wise_label==seg_idx)<=th_pix:
                score_mask[detect_seg_wise_label==seg_idx]=0
            else:
                if np.sum(gt_bool[detect_seg_wise_label==seg_idx])>0:
                    if full_img ==True and np.sum(detect_seg_wise_label==seg_idx)>np.prod(score_mask.shape)*0.5:
                        seg_fp = 1000
                    else:
                        pass
                else: 
                    seg_fp += 1
        if seg_fp>0:
            if len(image_dirs)>2:
                append_segment_img(bbox_info, seg_fp, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fp')
            else:
                pass
            
        defect_seg_num, defect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(gt_bool)
        # false negative or true positive
        for seg_idx in range(1, defect_seg_num):
            if np.sum(score_mask[defect_seg_wise_label==seg_idx])>0:
                if np.sum(score_mask[defect_seg_wise_label==seg_idx])<=th_pix:
                    score_mask[detect_seg_wise_label==seg_idx]=0
                else:
                    seg_tp += 1
            else: 
                seg_fn += 1
        if seg_tp>0:
            if len(image_dirs)>2:
                append_segment_img(bbox_info, seg_tp, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'tp')
            else:
                pass

        if seg_fn>0:
            if len(image_dirs)>2:
                append_segment_img(bbox_info, seg_fn, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fn')
            else:
                pass

    return cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, [seg_tp, seg_fn, seg_fp] 


# append a visualized image 
def append_segment_img(bbox_info, seg_idx, full_img, H, W, bgr_test_imgs, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, detect_result):
    if full_img ==True:
        y, x, width, height, area=bbox_info[seg_idx,:]
        offset_h, offset_w = int(height*0.2), int(width*0.2)
        if height > H or width >W :
            first_x, first_y = max(0,x-offset_h), max(0, y-offset_w)
            last_x, last_y=min(x+height+offset_h, score_mask.shape[-2]), min(y+width+offset_w, score_mask.shape[-1])
        else:
            first_x, first_y = max(0,x+height//2-H//2), max(0, y+width//2-W//2)
            last_x, last_y=min(first_x+H, score_mask.shape[-2]), min(first_y+W, score_mask.shape[-1])
    else:
        first_x, first_y =0, 0
        last_x, last_y =score_mask.shape[-2], score_mask.shape[-1]

    if len(bgr_test_imgs)>1:
        cropped_img=np.concatenate((bgr_test_imgs[0][first_x:last_x, first_y:last_y, :], bgr_test_imgs[1][first_x:last_x, first_y:last_y, :]), axis=1)
    else:
        cropped_img=bgr_test_imgs[0][first_x:last_x, first_y:last_y, :]


    if len(bgr_test_preds)==0:
        cropped_preds = None 
    elif len(bgr_test_preds)==1:
        cropped_preds = bgr_test_preds[0][first_x:last_x, first_y:last_y, :]
    else:
        cropped_preds_ = []
        for bgr_test_pred in bgr_test_preds:
            cropped_preds_.append(bgr_test_pred[first_x:last_x, first_y:last_y, :])
        cropped_preds = np.concatenate(cropped_preds_, axis=1) 
    cropped_gt=gt_bgr[first_x:last_x, first_y:last_y,:]
    cropped_score=score_mask[first_x:last_x, first_y:last_y]
    cropped_smaps = []
    for test_smap_bgr in test_smap_bgr_list:
        cropped_smaps.append(test_smap_bgr[first_x:last_x, first_y:last_y, :])
    cropped_smap=np.concatenate(cropped_smaps, axis = 1)
    
    cropped_fmaps = []
    for i in range(len(test_fmap_bgr)):
        cropped_fmaps.append(test_fmap_bgr[i][first_x:last_x, first_y:last_y, :])

    if len(cropped_fmaps)==0:
        cropped_feat_map = None
    else:
        cropped_feat_map = np.concatenate(cropped_fmaps, axis=1) 
    
    cropped_img_list.append(cropped_img)
    cropped_preds_list.append(cropped_preds)
    score_mask_list.append(cropped_score)
    score_map_list.append(cropped_smap)
    gt_mask_list.append(cropped_gt)
    feat_map_list.append(cropped_feat_map)
    file_list.append(os.path.join(os.path.join(image_dirs, detect_result), f'{file_name}~({first_x},{first_y},{last_x},{last_y})')) 


# save pixel-wise PR curves of two models that you desire to compare
def compare_scores(c, recall_list, precision_list, annot_list, detect_type, save_dir): 
    print('Exporting Recall Precision curve...')
    image_dirs = os.path.join(os.path.join(save_dir, 'stats'))
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    color_list = ['c', 'm', 'y', 'r', 'g', 'b']
    plt.title(f'Recall Precision Curve')
    plt.plot([1,0], ls="--")
    plt.plot([1,1], [1,0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')

    for i in range(len(recall_list)):
        recall = recall_list[i]
        precision = precision_list[i]
        auc_pr = auc(recall, precision)
        annot = f'{annot_list[i]} aupr: {auc_pr*100:0.2f}' 
        plt.plot(recall, precision, color = color_list[i], label = annot, linewidth = 1, alpha =0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    image_file = os.path.join(image_dirs, 'PR_curve_'+ detect_type + '_' + c.run_date.replace(':','') +'.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()



# save pixel-wise PR curve of a model
def save_scores(c, recall, precision, f1, thresholds, txt_file, detect_type, save_dir): 
    print('Exporting Recall Precision curve...')
    image_dirs = os.path.join(os.path.join(save_dir, 'stats'))
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)

    opt_idx = np.argmax(f1)
    optimized_th = thresholds[opt_idx]
    optimized_recall = recall[opt_idx]
    optimized_pr = precision[opt_idx]
    optimized_f1 = f1[opt_idx]
    auc_pr = auc(recall, precision)

    txt_file.write(f'\n\n{detect_type.upper()} Results')
    txt_file.write(f'\nBest F1 Score: {optimized_f1*100.0: .2f}')
    txt_file.write(f'\nAUC of PR curve: {auc_pr*100.0:0.2f}')
    txt_file.write(f'\nThreshold: {optimized_th:0.2f}, RECALL: {optimized_recall*100.0:.2f}, PRECISION: {optimized_pr*100.0:.2f}')

    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    plt.title(f'Recall Precision Curve')
    plt.plot(recall, precision)
    plt.scatter(optimized_recall, optimized_pr)
    plt.annotate(f'Threshold: {optimized_th:0.2f}\nRECALL: {optimized_recall*100.0:.2f}\nPRECISION: {optimized_pr*100.0:.2f}', xy=(optimized_recall, optimized_pr))
    plt.plot([1,0], ls="--")
    plt.plot([1,1], [1,0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')
    image_file = os.path.join(image_dirs, 'PR_curve_'+ detect_type + '_' + c.run_date.replace(':','')+'.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    return optimized_th
