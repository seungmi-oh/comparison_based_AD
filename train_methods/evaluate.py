import math
import numpy as np
import torch, copy
import torch.nn.functional as F
from visualize import *
from skimage import measure
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops


def _map_prob_exp(log_s, train_max):
    # p = exp(log_s - train_max)
    return torch.exp(log_s - train_max)

def _map_prob_exp_std(log_s, train_max, sigma_train):
    # p = exp((log_s - train_max)/sigma_train)
    return torch.exp((log_s - train_max) / (sigma_train + 1e-6))

def _map_prob_gauss_cdf(log_s, mu_train, sigma_train):
    # p = Phi((log_s - mu_train)/sigma_train)
    z = (log_s - mu_train) / (sigma_train + 1e-6)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def _pick_mapping(name):
    name = (name or "exp").lower()
    assert name in {"exp", "exp_std", "gauss_cdf"}
    return name

# Calculate anomaly scores
class Anomaly_Score_Calculator:
    def __init__(self, pool_layers, inp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, infer_type, pro, w_fe =0, best_w_fe =False):
        self.pool_layers = pool_layers
        self.inp_size = inp_size
        self.height = height
        self.width = width
        self.train_dist = train_dist
        self.train_gt_mask_list = train_gt_mask_list
        self.pro = pro

        if len(train_pred_list)==2:
            self.train_fe_super_mask = np.array(train_pred_list[-1], dtype=np.float32)
        elif len(train_pred_list)==0:
            pass
        else:
            self.train_fe_super_mask = np.array(train_pred_list, dtype=np.float32)

        self.infer_type = infer_type
        self.best_w_fe = best_w_fe
        self.w_fe = w_fe
    
    
    def get_train_NFs_anomaly_score_map(self, prob_mapping: str = None, inverting_mode: str = 'layerwise'):
        """
        학습/통계 수집 + train 슈퍼마스크 생성:
        - exp/exp_std: train_max 사용
        - gauss_cdf  : train μ, σ 사용
        - 레이어별 p_l 업샘플 → 합산(score_map) → 마지막에만 뒤집기
            self.train_nf_super_mask = score_map.max() - score_map
        """
        self.prob_mapping = _pick_mapping(prob_mapping or getattr(self, "prob_mapping", "exp"))
        mapping = self.prob_mapping

        self.train_max_list = []
        self.layer_mu  = []
        self.layer_std = []

        train_maps = []

        for l, _ in enumerate(self.pool_layers):
            train_log = torch.as_tensor(self.train_dist[l], dtype=torch.float32)  # (B,*)

            # 통계 저장
            m  = train_log.max()
            mu = train_log.mean()
            sd = train_log.std(unbiased=True) + 1e-6

            self.train_max_list.append(float(m))
            self.layer_mu.append(float(mu))
            self.layer_std.append(float(sd))

            # 레이어별 확률 p_l
            if mapping == "exp":
                p = _map_prob_exp(train_log, m)
            elif mapping == "exp_std":
                p = _map_prob_exp_std(train_log, m, sd)
            else:  # "gauss_cdf"
                p = _map_prob_gauss_cdf(train_log, mu, sd)

            mask = p.reshape(-1, self.height[l], self.width[l])  # (B,H_l,W_l)
            up = F.interpolate(mask.unsqueeze(1), size=self.inp_size,
                            mode='bilinear', align_corners=True).squeeze(1)  # (B,H,W)
            train_maps.append(up.detach().cpu().numpy())

        # 레이어 합산
        if inverting_mode == 'layerwise':
            score_map = np.zeros_like(train_maps[0])
            for m in train_maps:
                score_map += m.max() - m
            self.train_nf_super_mask = score_map  # (B,H,W)
        else:
            score_map = np.zeros_like(train_maps[0])
            for m in train_maps:
                score_map += m
            self.train_nf_super_mask = score_map.max() - score_map  # (B,H,W)

        # GT 마스크 정리
        train_gt_mask = np.asarray(self.train_gt_mask_list, dtype=np.float32)
        if len(train_gt_mask.shape) == 4:
            self.train_gt_mask = np.asarray(np.mean(train_gt_mask, axis=1), dtype=np.bool_)
        else:
            self.train_gt_mask = np.asarray(train_gt_mask, dtype=np.bool_)

    # ================== TEST ==================
    def get_NFs_anomaly_score_map(self, test_dist, prob_mapping: str = None, inverting_mode: str = 'layerwise'):
        """
        테스트 단계:
        - 선택 매핑(exp/exp_std: train_max, gauss_cdf: μ,σ)
        - 레이어별 p_l 업샘플 → 합산(score_map) → 마지막에만 뒤집기
        """
        assert hasattr(self, "train_max_list") and hasattr(self, "layer_mu") and hasattr(self, "layer_std"), \
            "run get_train_NFs_anomaly_score_map() first"

        mapping = _pick_mapping(prob_mapping or getattr(self, "prob_mapping", "exp"))
        test_maps = []

        for l, _ in enumerate(self.pool_layers):
            test_log = torch.as_tensor(test_dist[l], dtype=torch.float32)

            train_max = torch.as_tensor(self.train_max_list[l], dtype=torch.float32)
            mu  = torch.as_tensor(self.layer_mu[l],  dtype=torch.float32)
            sd  = torch.as_tensor(self.layer_std[l], dtype=torch.float32)

            if mapping == "exp":
                p = _map_prob_exp(test_log, train_max)
            elif mapping == "exp_std":
                p = _map_prob_exp_std(test_log, train_max, sd)
            else:  # "gauss_cdf"
                p = _map_prob_gauss_cdf(test_log, mu, sd)

            mask = p.reshape(-1, self.height[l], self.width[l])
            up = F.interpolate(mask.unsqueeze(1), size=self.inp_size,
                            mode='bilinear', align_corners=True).squeeze(1)
            test_maps.append(up.detach().cpu().numpy())

        # 레이어 합산
        if inverting_mode == 'layerwise':
            score_map = np.zeros_like(test_maps[0])
            for m in test_maps:
                score_map += m.max() - m
            nf_super_mask = score_map  # (B,H,W)
        else:
            score_map = np.zeros_like(test_maps[0])
            for m in test_maps:
                score_map += m
            nf_super_mask = score_map.max() - score_map  # (B,H,W)
        return nf_super_mask

    # ================== (옵션) 개별 레이어 ==================
    def get_indivdidual_NF_anomaly_score_map(self, test_dist, l: int, prob_mapping: str = None):
        """
        단일 레이어 anomaly map:
        - p_l 업샘플 → layer 내에서 한 번만 뒤집기 (max(p_l) - p_l) 반환
        - 전체 합산의 최종 뒤집기와 별개로, 레이어 단위 결과를 보고 싶을 때 사용
        """
        assert hasattr(self, "train_max_list") and hasattr(self, "layer_mu") and hasattr(self, "layer_std"), \
            "run get_train_NFs_anomaly_score_map() first"

        mapping = _pick_mapping(prob_mapping or getattr(self, "prob_mapping", "exp"))

        test_log = torch.as_tensor(test_dist[l], dtype=torch.float32)
        train_max = torch.as_tensor(self.train_max_list[l], dtype=torch.float32)
        mu  = torch.as_tensor(self.layer_mu[l],  dtype=torch.float32)
        sd  = torch.as_tensor(self.layer_std[l], dtype=torch.float32)

        if mapping == "exp":
            p = _map_prob_exp(test_log, train_max)
        elif mapping == "exp_std":
            p = _map_prob_exp_std(test_log, train_max, sd)
        else:  # "gauss_cdf"
            p = _map_prob_gauss_cdf(test_log, mu, sd)

        mask = p.reshape(-1, self.height[l], self.width[l])
        up = F.interpolate(mask.unsqueeze(1), size=self.inp_size,
                        mode='bilinear', align_corners=True).squeeze(1)
        # 레이어 내 invert (진짜 최종과 동일한 스케일은 아님)
        a_layer = (up.max().item()) - up.detach().cpu().numpy()
        return a_layer
   
    def get_pix_anomaly_score_map(self, pred_list):
        if len(pred_list)==2:
            fe_super_mask = np.array(pred_list[-1], dtype=np.float32)
        else:
            fe_super_mask = np.array(pred_list, dtype=np.float32)
        return fe_super_mask

    def aggregate_anomaly_score_map(self, fe_super_mask, nf_super_mask, w_fe):
        joint_super_mask = copy.deepcopy((1-w_fe)*(nf_super_mask/len(self.pool_layers)))
        joint_super_mask += w_fe*fe_super_mask
        return joint_super_mask

    # find the best weight using grid search
    def get_best_w_fe(self, fe_super_mask, nf_super_mask, gt_mask, is_full =False, test_dataset=None, full_dims=None):
        best_w_fe = 0
        best_aupr = 0
        gt_mask = np.uint8(gt_mask.sum(1)>0)
        aupr_list = []
        if is_full ==True:
            if type(gt_mask) is list:
                gt_mask= make_full_chip([gt_mask], test_dataset.chip_list, test_dataset.x, full_dims, gt_mask[0].shape[-2], gt_mask[0].shape[-1]) 
            elif type(gt_mask) is np.ndarray:
                gt_mask= make_full_chip([gt_mask], test_dataset.chip_list, test_dataset.x, full_dims, gt_mask.shape[-2], gt_mask.shape[-1]) 
            else: 
                raise NotImplementedError(f'{type(gt_mask)} is not implemented for gt_mask type')
        else:
            pass
        for w in range(1,20):
            w_fe = w*0.05
            joint_super_mask = self.aggregate_anomaly_score_map(fe_super_mask, nf_super_mask, w_fe)
            
            if is_full ==True:
                joint_super_mask= make_full_chip([joint_super_mask], test_dataset.chip_list, test_dataset.x, full_dims, joint_super_mask.shape[-2], joint_super_mask.shape[-1]) 
                Y = joint_super_mask[0].flatten()
                Y_label = gt_mask[0].flatten()
            else:
                Y = joint_super_mask.flatten()
                Y_label = gt_mask.flatten()

            precision, recall, thresholds = precision_recall_curve(Y_label, Y)
            auc_pr = auc(recall, precision)
            aupr_list.append([w_fe, auc_pr])
            if auc_pr == max(auc_pr, best_aupr):
                best_w_fe = w_fe
                best_aupr = auc_pr
            else:
                pass
        return best_w_fe, aupr_list


    def save_results(self, c, epoch, test_image_list, test_dist, pred_list, gt_mask_list, gt_label_list, feat_maps_list, test_dataset, full_dims, det_roc_obs, seg_roc_obs, seg_pr_obs, save_dir, txt_file, result_dict, is_full =False, eval_metrics = True, verbose=True, indiv_pr_obs=[], prob_mapping= 'exp', inverting_mode='layerwise'):
        self.gt_mask = np.asarray(gt_mask_list, dtype=np.bool)
        if 'fe_only' in self.infer_type:
            self.super_mask = self.get_pix_anomaly_score_map(pred_list)
            if is_full ==True:
                self.test_img_full, self.super_mask_full, self.gt_mask_full, self.fe_super_mask_full, self.feat_maps_full= make_full_chip([test_image_list, self.super_mask, self.gt_mask, pred_list, feat_maps_list], test_dataset.chip_list, test_dataset.x, full_dims, self.super_mask.shape[-2], self.super_mask.shape[-1]) 
            else:
                self.gt_mask_full, self.super_mask_full, self.test_img_full, self.fe_super_mask_full, self.feat_maps_full = self.gt_mask, self.super_mask, test_image_list, pred_list, feat_maps_list
        elif 'joint' in self.infer_type:
            self.fe_super_mask = self.get_pix_anomaly_score_map(pred_list)
            self.get_train_NFs_anomaly_score_map(prob_mapping, inverting_mode)
            self.nf_super_mask = self.get_NFs_anomaly_score_map(test_dist, prob_mapping, inverting_mode)
            if self.best_w_fe == True:
                self.w_fe, self.aupr_list = self.get_best_w_fe(self.fe_super_mask, self.nf_super_mask, self.gt_mask, is_full, test_dataset, full_dims)
                # 기존 결과 불러오기
                if os.path.isdir(save_dir)==False:
                    os.makedirs(save_dir)
                with pd.ExcelWriter(os.path.join(save_dir, 'aupr_results.xlsx'), engine="openpyxl", mode="w") as writer:
                    # aupr_list를 dict list로 변환
                    aupr_dicts = [{"w_fe": w, "aupr": a} for w, a in self.aupr_list]

                    # DataFrame으로 변환
                    df = pd.DataFrame(aupr_dicts)

                    # 새로운 시트에 쓰기 (예: "AUPR_Result")
                    df.to_excel(writer, sheet_name="AUPR_Result", index=False)
                result_dict['best_w_fe'] = self.w_fe
            else:
                pass
            self.super_mask = self.aggregate_anomaly_score_map(self.fe_super_mask, self.nf_super_mask, self.w_fe)
            if is_full ==True:
                self.test_img_full, self.super_mask_full, self.gt_mask_full, self.fe_super_mask_full, self.feat_maps_full= make_full_chip([test_image_list, self.super_mask, self.gt_mask, pred_list, feat_maps_list], test_dataset.chip_list, test_dataset.x, full_dims, self.super_mask.shape[-2], self.super_mask.shape[-1]) 
            else:
                self.gt_mask_full, self.super_mask_full, self.test_img_full, self.fe_super_mask_full, self.nf_super_mask_full, self.feat_maps_full = self.gt_mask, self.super_mask, test_image_list, pred_list, self.nf_super_mask, feat_maps_list
            self.train_super_mask = self.aggregate_anomaly_score_map(self.train_fe_super_mask, self.train_nf_super_mask, self.w_fe)
        else:
            self.get_train_NFs_anomaly_score_map(prob_mapping, inverting_mode)
            indiv_super_masks = []
            if len(indiv_pr_obs) > 0:
                for l, p in enumerate(self.pool_layers):
                    super_mask_l = self.get_indivdidual_NF_anomaly_score_map(test_dist, l, prob_mapping)  # (B,H,W)
                    indiv_super_masks.append(super_mask_l)

            # 종합 anomaly map
            if inverting_mode == 'layerwise' and len(indiv_pr_obs) > 0:
                self.super_mask = np.mean(np.stack(indiv_super_masks, axis=0), axis=0)  # (B,H,W)
            else:
                self.super_mask = self.get_NFs_anomaly_score_map(test_dist, prob_mapping, inverting_mode)

            self.train_super_mask = self.train_nf_super_mask
            if is_full == True:
                # 종합
                (self.test_img_full, self.super_mask_full, self.gt_mask_full,
                self.fe_super_mask_full, self.feat_maps_full) = make_full_chip(
                    [test_image_list, self.super_mask, self.gt_mask, pred_list, feat_maps_list],
                    test_dataset.chip_list, test_dataset.x, full_dims,
                    self.super_mask.shape[-2], self.super_mask.shape[-1]
                )
                # 개별
                indiv_super_mask_full_list = []
                if len(indiv_pr_obs) > 0:
                    for super_mask_l in indiv_super_masks:
                        _, super_mask_full_l, _, _, _ = make_full_chip(
                            [test_image_list, super_mask_l, self.gt_mask, pred_list, feat_maps_list],
                            test_dataset.chip_list, test_dataset.x, full_dims,
                            super_mask_l.shape[-2], super_mask_l.shape[-1]
                        )
                        indiv_super_mask_full_list.append(super_mask_full_l)
            else:
                # 종합
                self.gt_mask_full, self.super_mask_full = self.gt_mask, self.super_mask
                self.test_img_full, self.fe_super_mask_full, self.feat_maps_full = test_image_list, pred_list, feat_maps_list
                # 개별
                indiv_super_mask_full_list = indiv_super_masks if len(indiv_pr_obs) > 0 else []

        self.score_label = np.max(self.super_mask, axis=(1, 2))
        self.gt_label = np.asarray(gt_label_list, dtype=np.bool)
        gt_mask_full = np.asarray(self.gt_mask_full, dtype=np.bool)
        if gt_mask_full.shape[1]>0 and len(gt_mask_full.shape)==4:
            gt_mask_full = (gt_mask_full.sum(1)>0)
        else:
            pass
        self.gt_mask_bool_full = gt_mask_full

        if eval_metrics == False:
            pass
        else:
            #AUROC at the image-level
            if det_roc_obs is None:
                pass
            else:
                det_roc_auc = roc_auc_score(self.gt_label, self.score_label)
                _ = det_roc_obs.update(100.0*det_roc_auc, epoch, txt_file, verbose)
                if len(save_dir)>0:
                    det_precision, det_recall, det_thresholds = precision_recall_curve(self.gt_label, self.score_label)
                    a = 2 * det_precision * det_recall
                    b = det_precision + det_recall
                    det_f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                    det_threshold = save_scores(c, det_recall, det_precision, det_f1, det_thresholds, txt_file, 'det', save_dir)
                    export_hist(c, self.gt_label, self.score_label, 'det', save_dir)
                if verbose ==True:
                    result_dict[det_roc_obs.name] = 100.0*det_roc_auc 

            #AUPR at the pixel-level
            
            pix_precision, pix_recall, pix_ths = precision_recall_curve(gt_mask_full.flatten(), self.super_mask_full.flatten())
            seg_pr_auc = auc(pix_recall, pix_precision)
            save_best_seg_weights = seg_pr_obs.update(100.0 * seg_pr_auc, epoch, txt_file, verbose)
            if len(save_dir) > 0:
                a = 2 * pix_precision * pix_recall
                b = pix_precision + pix_recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                seg_threshold = save_scores(c, pix_recall, pix_precision, f1, pix_ths, txt_file, 'pix', save_dir)
                export_hist(c, gt_mask_full.flatten(), self.super_mask_full.flatten(), 'pix', save_dir)
                result_dict['threshold'] = seg_threshold
            if verbose == True:
                result_dict[seg_pr_obs.name] = 100.0 * seg_pr_auc

            # ===== AUPR (pixel-level) : 개별 NF =====
            if len(indiv_pr_obs) > 0 and len(indiv_super_mask_full_list) == len(indiv_pr_obs):
                for li, super_full_l in enumerate(indiv_super_mask_full_list):
                    p_l, r_l, _ = precision_recall_curve(gt_mask_full.flatten(), super_full_l.flatten())
                    aupr_l = auc(r_l, p_l)
                    indiv_pr_obs[li].update(100.0 * aupr_l, epoch, txt_file, verbose)
                    result_dict[indiv_pr_obs[li].name] = 100.0 * aupr_l

            # AUROC at the pixel-level
            if seg_roc_obs is None:
                pass
            else:
                seg_roc_auc = roc_auc_score(gt_mask_full.flatten(), self.super_mask_full.flatten())
                save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch, txt_file, verbose)
                if verbose ==True:
                    result_dict[seg_roc_obs.name] = 100.0*seg_roc_auc 

            #AUPRO 
            if self.pro ==True:
                txt_file_fname = txt_file.name
                txt_file_seg_fname = txt_file_fname.replace('.txt', '_seg_th.txt')
                seg_pro, opt_th = cal_pro_metric(gt_mask_full, self.super_mask_full, txt_file_seg_fname)
                if verbose ==True:
                    print(f'    AUPRO: \t max: {seg_pro*100:.2f}')
                    print(f'optimal threshold from AUPRO: {opt_th:0.4f}')
                    txt_file.write(f'\n    AUPRO: \t max: {seg_pro*100:.2f}')
                    txt_file.write(f'\noptimal threshold from AUPRO: {opt_th:0.4f}')
                    result_dict['AUPRO'] = 100.0*seg_pro
                    result_dict['th_AUPRO'] = opt_th 

        if is_full ==True:
            self.test_img_crop, self.super_mask_crop, self.gt_mask_crop, self.fe_super_mask_crop, self.feat_maps_crop = crop_full_chip([self.test_img_full, self.super_mask_full, self.gt_mask_full, self.fe_super_mask_full, self.feat_maps_full], test_dataset.chip_list, test_dataset.x, full_dims, self.super_mask.shape[-2], self.super_mask.shape[-1])
            if self.gt_mask_crop.shape[1]>0 and len(self.gt_mask_crop.shape)==4:
                gt_mask_crop = (self.gt_mask_crop.sum(1)>0)
            else:
                pass
            self.gt_mask_bool_crop = gt_mask_crop
        else:
            self.test_img_crop, self.super_mask_crop, self.gt_mask_crop, self.fe_super_mask_crop, self.feat_maps_crop = self.test_img_full, self.super_mask_full, self.gt_mask_full, self.fe_super_mask_full, self.feat_maps_full
            if self.gt_mask_crop.shape[1]>0 and len(self.gt_mask_crop.shape)==4:
                gt_mask_crop = (self.gt_mask_crop.sum(1)>0)
            else:
                pass
            self.gt_mask_bool_crop = gt_mask_crop
        return result_dict


def plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, compare_mode, base_anomaly_cal=None, add_fe_anomaly=None):
    if add_fe_anomaly is None:
        add_fe_anomaly = c.add_fe_anomaly
    image_dirs = os.path.join(save_dir, 'stats')
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    # save histogram plot with anomaly scores of training set
    if compare_mode == 'normal':
        Y = anomaly_cal.super_mask_crop.flatten()
        Y_label = anomaly_cal.gt_mask_bool_crop.flatten()
        defect_num = np.sum(Y_label==1)
        Y_train = anomaly_cal.train_super_mask.flatten() 
        plt.hist([Y_train, Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['b', 'r', 'g'], label=['TRAIN', 'ANO', 'TYP'], alpha=0.3, histtype='stepfilled')
        image_file = os.path.join(image_dirs, f'hist_plots_pix_with_trainset_' + c.run_date.replace(':', '')+'.png')
    # save histogram plots to compare the prediction of CNF networks and the combined prediction 
    elif compare_mode == 'score_aggregation':
        Y = anomaly_cal.super_mask_crop.flatten()
        Y_label = gt_mask_bool_crop.flatten()
        defect_num = np.sum(Y_label==1)
        nf_Y = anomaly_cal.nf_super_mask_crop.flatten()
        plt.hist([nf_Y[Y_label==1], nf_Y[Y_label==0],Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['tab:purple', 'tab:cyan', 'tab:red', 'tab:green'], label=['ANO(nf)', 'TYP(nf)', 'ANO(nf+cls)', 'TYP(nf+cls)'], alpha=0.3, histtype='stepfilled')
        comp_precision, comp_recall, _= precision_recall_curve(Y_label, nf_Y)
        precision, recall, _= precision_recall_curve(Y_label, Y)
        compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['nf', 'nf+cls'], f'pix_compare_{compare_mode}', save_dir)
        image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_' + c.run_date.replace(':', '')+'.png')
    # save histogram plots to compare the finetuned feature extractor and the pretrained feature extractor 
    elif compare_mode == 'finetune_encoder':
        w_fe_base = base_anomaly_cal.w_fe
        w_fe = anomaly_cal.w_fe
        Y = anomaly_cal.super_mask.flatten()
        Y_label = anomaly_cal.gt_mask.flatten()
        defect_num = np.sum(Y_label==1)
        base_Y = base_anomaly_cal.super_mask.flatten()
        plt.hist([base_Y[Y_label==1], base_Y[Y_label==0],Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['tab:purple', 'tab:cyan', 'tab:red', 'tab:green'], label=['ANO(pretrained)', 'TYP(pretrained)', 'ANO(finetuned)', 'TYP(finetuned)'], alpha=0.3, histtype='stepfilled')

        comp_precision, comp_recall, _= precision_recall_curve(Y_label, base_Y)
        precision, recall, _= precision_recall_curve(Y_label, Y)
        if add_fe_anomaly ==True:
            compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['pre-trained', 'fine-tuned'], f'pix_compare_{compare_mode}-pretrain_{w_fe_base:0.2f}-finetune_{w_fe:0.2f}', save_dir)
            image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_base_{w_fe_base:0.2f}_proposed_{w_fe:0.2f}_' + c.run_date.replace(':', '')+'.png')
        else:
            compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['pre-trained', 'fine-tuned'], f'pix_compare_{compare_mode}', save_dir)
            image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_' + c.run_date.replace(':', '')+'.png')
    else:
        NotImplementedError(f'{compare_mode} is not implemented for compare_mode')

    plt.legend(loc = 'center left', bbox_to_anchor =(1,0.5))
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


def cal_segmentwise_eval_results(c, gt_mask, super_mask, threshold):
    seg_tp = 0
    seg_fp = 0
    seg_fn = 0
    for i in range(len(gt_mask)):
        # gts
        gt = gt_mask[i].astype(np.float32)

        if len(gt.shape)==3 and gt.shape[0]>1:
            gt = np.sum(gt, 0)
            gt = np.uint8(gt>0)
        else:
            pass

        gt = (255.0*gt).astype(np.uint8)
        dilate_kernel = np.ones((3,3), dtype=np.uint8)
        gt = cv2.dilate(gt, dilate_kernel, iterations=1)

        if c.is_k_disk==True:
            kernel = morphology.disk(c.k_size)
        else:
            kernel = np.ones(c.k_size)

        # anomaly prediction
        score_mask = np.zeros_like(super_mask[i])
        score_mask[super_mask[i] >=  threshold] = 1.0
        score_mask = (255.0*score_mask).astype(np.uint8)
        if c.is_close == True:
            score_mask = morphology.closing(score_mask, kernel)
        if c.is_open == True:
            score_mask = morphology.opening(score_mask, kernel)
        score_mask = (255.0*score_mask).astype(np.uint8)

        detect_seg_num, detect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(score_mask)
        for seg_idx in range(1, detect_seg_num):
            if np.sum(detect_seg_wise_label==seg_idx)<=c.th_pix:
                score_mask[detect_seg_wise_label==seg_idx]=0
            else:
                if np.sum(gt[detect_seg_wise_label==seg_idx])>0:
                    if np.sum(detect_seg_wise_label==seg_idx)>np.prod(score_mask.shape)*0.5:
                        seg_fp = 1000
                    else:
                        pass
                else: 
                    seg_fp += 1

        defect_seg_num, defect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(gt)
        # false negative or true positive
        for seg_idx in range(1, defect_seg_num):
            if np.sum(score_mask[defect_seg_wise_label==seg_idx])>0:
                if np.sum(score_mask[defect_seg_wise_label==seg_idx])<=c.th_pix:
                    score_mask[detect_seg_wise_label==seg_idx]=0
                    seg_fn += 1
                else:
                    seg_tp += 1
            else: 
                seg_fn += 1

    return  seg_tp, seg_fn, seg_fp

# visualize inference results
def get_threshold_with_fixed_fn(c, super_mask, gt_mask, save_dir, class_name):
    max_score = np.max(super_mask)
    min_score = np.min(super_mask)

    sample_num = 30
    step = (max_score-min_score)/sample_num
    print('Max score')
    print(max_score)

    print('Min score')
    print(min_score)

    txt_file_th = open(os.path.join(save_dir, f'threshold_with_fn-{c.fn}_th_pix-{c.th_pix}.txt'), 'w')
    csv_path_th = os.path.join(save_dir, f'threshold_with_fn-{c.fn}_th_pix-{c.th_pix}.csv')

    txt_file_th.write('Max score\n')
    txt_file_th.write(f'{max_score}\n')
    txt_file_th.write('Min score\n')
    txt_file_th.write(f'{min_score}\n')

    idx=1
    while idx <= sample_num:
        threshold = max_score - step*idx

        seg_tp, seg_fn, seg_fp = cal_segmentwise_eval_results(c, gt_mask, super_mask, threshold)

        recall_ = seg_tp/(seg_tp + seg_fn) 
        precision_ = seg_tp/(seg_tp+seg_fp+1e-7)
        f1_ = 2*recall_*precision_/(recall_+precision_+1e-7)
        print(f'{idx}/{sample_num}')
        print(f'Threshold:{threshold:0.4f} Recall: {recall_*100:0.2f} ({seg_tp}/{seg_tp+seg_fn}), Precision: {precision_*100: 0.2f} ({seg_tp}/{seg_tp+seg_fp}), f1: {f1_*100: 0.2f}')
        txt_file_th.write(f'{idx}/{sample_num}\n')
        txt_file_th.write(f'Threshold:{threshold:0.4f} Recall: {recall_*100:0.2f} ({seg_tp}/{seg_tp+seg_fn}), Precision: {precision_*100: 0.2f} ({seg_tp}/{seg_tp+seg_fp}), f1: {f1_*100: 0.2f}\n')
        print(c.fn)
        print(seg_fn)
        if seg_fn > c.fn:
            idx = idx+1
        else:
            if (max_score-min_score)<0.0001:
                seg_result = {'threshold': threshold, 'recall': f'{recall_*100:0.2f} ({seg_tp}/{seg_tp+seg_fn})', 'precision': f'{precision_*100:0.2f} ({seg_tp}/{seg_tp+seg_fp})', 'f1': f'{f1_*100:0.2f}'}
                break
            else:
                idx = 1
                min_score = threshold 
                max_score = threshold + step 

                sample_num = 10
                step = (max_score-min_score)/sample_num

                print(f'\nInitialize max score as {max_score} and min score as {min_score}')
                txt_file_th.write(f'\nInitialize max score as {max_score} and min score as {min_score}\n')

    txt_file_th.close()
    keys = list(seg_result.keys())
    df_result = pd.DataFrame(seg_result, index = [class_name])
    df_result.to_csv(csv_path_th, header=True, float_format = '%.4f')
    return threshold


# visualize inference results
def viz(c, data_cfg, test_loader, anomaly_cal, txt_file, save_dir, result_dict, w_fe=0):
    test_dataset = test_loader.dataset

#    if c.is_full ==True:
#        file_list = test_dataset.chip_list
#    else:
    file_list = test_dataset.x

    if c.th_manual >0:
        seg_threshold = c.th_manual 
        result_dict['threshold'] = seg_threshold
        print(f'Manual PIX Threshold: {seg_threshold:0.4f}')
    else:
        seg_threshold = result_dict['threshold']
        print(f'Optimal PIX Threshold: {seg_threshold:0.4f}')
    # set directory
    dir_feature=['viz']
    if c.th_pix >=0:
        dir_feature.append(f'{c.th_pix}')
    if c.add_fe_anomaly==True:
        dir_feature.append(f'{w_fe:0.2f}')
    if c.is_open==True:
        dir_feature.append('imopen')
    if c.is_close==True:
        dir_feature.append('imclose')
    if c.fn >=0:
        dir_feature.append(f'fixed_fn_{c.fn}')
    if c.th_manual>0:
        dir_feature.append(f'manual_th_{c.th_manual:0.4f}')
    if c.save_all_tn_sample ==True:
        dir_feature.append('save_all_tn')
    if c.is_full ==True:
        dir_feature.append('full_chip')
    else:
        dir_feature.append('cropped_img')

    str_dir_features = '-'.join(dir_feature)
    image_dirs = os.path.join(save_dir, str_dir_features)
    makedirs(image_dirs)

    seg_performance = cal_segmentwise_eval_results(c, anomaly_cal.gt_mask_bool_full, anomaly_cal.super_mask_full, seg_threshold)
    tp, fn, fp = seg_performance
    recall = tp/(fn+tp+1e-7)
    precision = tp/(fp+tp+1e-7)
    f1_score = 2*recall*precision/(recall+precision+1e-7)
    txt_file.write(f'\nTP num: {tp}, FN num: {fn}, FP num: {fp}')
    txt_file.write(f'\nSEG Performance) recall: {recall*100.0 :0.2f}, precision: {precision*100.0:0.2f}, f1-score: {f1_score*100.0:0.2f}')
    result_dict['TP'] = tp 
    result_dict['FN'] = fn
    result_dict['FP'] = fp
    result_dict['recall'] = f'{recall*100.0:.2f} ({tp}/{tp+fn})'
    result_dict['precision'] = f'{precision*100.0:.2f} ({tp}/{tp+fp})'
    result_dict['f1-score'] = f1_score*100.0
    result_dict['fp_over_P'] = f'{(fp/(tp+fn+1e-8))*100.0:.2f} ({fp}/{tp+fn})'
    csv_path = os.path.join(image_dirs, 'test_result.csv') 
    write_csv(result_dict, csv_path, test_dataset.class_name)


    test_img = anomaly_cal.test_img_crop
    gt_mask = anomaly_cal.gt_mask_crop
    super_mask = anomaly_cal.super_mask_crop
    fe_super_mask = anomaly_cal.fe_super_mask_crop
    gt_label = anomaly_cal.gt_label
    score_label = anomaly_cal.score_label
    feat_maps = anomaly_cal.feat_maps_crop
    w_fe = anomaly_cal.w_fe

    if 'fe_only' in c.infer_type:
        export_test_images(c, data_cfg, test_img, gt_mask, super_mask, [], feat_maps, seg_threshold, file_list, image_dirs, txt_file, 0.0)
    else:
        export_test_images(c, data_cfg, test_img, gt_mask, super_mask, fe_super_mask, feat_maps, seg_threshold, file_list, image_dirs, txt_file, w_fe)
    txt_file.close()


# calculate AUPRO metric
# We modified the function of the CDO project (https://github.com/caoyunkang/CDO/tree/master)  
def cal_pro_metric(labeled_imgs, score_imgs, txt_file_seg_fname, fpr_thresh=0.3, fpr_thresh_seg = 0.1, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    defect_ratio = np.mean(labeled_imgs) 
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    recalls = []
    precisions = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    txt_file_seg = open(txt_file_seg_fname, 'w')
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1
        tp=0
        fp=0
        fn=0
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            if step > 0: 
                if fprs[step-1] <= fpr_thresh_seg:
                    score_map = measure.label(binary_score_maps[i].astype(np.bool), connectivity=2)
                    score_props = measure.regionprops(score_map)
                else:
                    pass
            else:
                pass
            tp_ = 0
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                if intersection>0:
                    tp_ += 1
                else:
                    fn += 1
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if step > 0: 
                if fprs[step-1] <= fpr_thresh_seg:
                    fp += len(score_props)-tp_
                    tp += tp_
                else:
                    pass
            else:
                pass
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        if step > 0: 
            if fprs[step-1] <= fpr_thresh_seg:
                txt_file_seg.write(f'\n{step}/{max_steps}, th:\t {thred} tp:\t {tp} fn:\t {fn} fp:\t {fp}')
                recalls.append(tp/(tp+fn))
                precisions.append(tp/(tp+fp+1e-8))
            else:
                pass
        else:
            pass
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    txt_file_seg.close()

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected_org = fprs[idx]
    fprs_selected = rescale(fprs_selected_org)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)

    idx_th = fprs <= fpr_thresh_seg  # find the indexs of fprs that is less than expect_fpr (default 0.1) 
    threds_selected = threds[idx_th][1:]
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    a = 2 * recalls * precisions 
    b = recalls + precisions
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    opt_idx = np.argmax(f1)
    opt_th = threds_selected[opt_idx]
    return pro_auc_score, opt_th

