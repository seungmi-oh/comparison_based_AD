'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, math
import numpy as np
import torch

__all__ = ('save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate', 'warmup_group_learning_rate', 'adjust_group_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# save test results for the last epoch
def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, model_dir, class_name, run_date, data_type = 'real'):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    result_dir = os.path.join(model_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fp = open(os.path.join(result_dir, f'{data_type}-{run_date}.txt'), "w")
    fp.write(result)
    fp.close()


# save a trained model
def save_weights(c, model, model_dir):
    weight_dir = os.path.join(model_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if c.finetuning == False:
        encoder = model[0]
        nfs = model[-1]
        state = {'encoder_state_dict': encoder.state_dict(),
                 'nf_state_dict': [nf.state_dict() for nf in nfs],
                 'args': c}
    else:
        encoder = model[0]
        decoder = model[1]
        if 'fe_only' in c.train_type:
            state = {'encoder_state_dict': encoder.state_dict(),
                     'decoder_state_dict': decoder.state_dict(),
                     'args': c}
        elif 'nf' in c.train_type:
            nfs = model[-1]
            state = {'encoder_state_dict': encoder.state_dict(),
                     'nf_state_dict': [nf.state_dict() for nf in nfs],
                     'args': c}
        else:
            raise NotImplementedError('{} is not supported train type!'.format(c.train_type))
    filename = f'{c.train_type}_{c.run_date}.pt'
    path = os.path.join(weight_dir, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(path))


# load weights according to inference type
def load_weights(c, model, infer_type):
    weight_dir = os.path.join(c.model_dir, 'weights')
    if 'joint' in infer_type:
        sub_path = infer_type.replace('joint', 'nf_only')
    else:
        sub_path = infer_type

    if os.path.exists(weight_dir)==True:
        model_files_=os.listdir(weight_dir)
        for model_file_ in model_files_:
            if (sub_path in model_file_)==True:
                model_path=os.path.join(weight_dir, model_file_)
            else: 
                pass
    if 'model_path' in locals():
        state = torch.load(model_path)
        c.num_class = state['args'].num_class
        if 'encoder_state_dict' in state:
            encoder = model[0]
            encoder.load_state_dict(state['encoder_state_dict'], strict=True)
        if 'nf_state_dict' in state:
            nfs = model[-1]
            nfs = [nf.load_state_dict(state, strict=True) for nf, state in zip(nfs, state['nf_state_dict'])]
        if 'decoder_state_dict' in state:
            decoder = model[1]
            decoder.load_state_dict(state['decoder_state_dict'], strict=True)
        print('Loading weights from {}'.format(model_path))
        return state
    else:
        print(f'No {infer_type} weights from {weight_dir}')
        return []


# learning rate scheduler
def adjust_learning_rate(c, optimizer, epoch, total_epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / total_epoch)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# warm up learning rate scheduler
def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate

def _ensure_base_lrs_and_warmup_lists_for_optims(c, optimizers, total_epoch=None):
    """
    optimizers (리스트)에 맞춰 base_lrs / warmup 리스트(c.lr_warmup_*_list)를 보장.
    - c.lr_nf_list가 있으면 그걸, 없으면 단일 c.lr을 복제해 사용.
    - 각 optimizer의 모든 param_group에 base_lr가 없다면 주입.
    """
    num = len(optimizers)

    # 1) base_lrs 결정
    if getattr(c, 'lr_nf_list', None):
        assert len(c.lr_nf_list) == num, \
            f"--lr_nf_list 길이({len(c.lr_nf_list)}) != optimizers({num})"
        base_lrs = [float(x) for x in c.lr_nf_list]
    else:
        base_lrs = [float(c.lr)] * num

    # 각 optimizer의 param_group에 base_lr 주입(없을 때만)
    for opt, blr in zip(optimizers, base_lrs):
        for g in opt.param_groups:
            if 'base_lr' not in g:
                g['base_lr'] = blr
            # base_lr가 있는데 lr가 비어있으면 lr도 일치시켜줌(안전장치)
            if 'lr' not in g or g['lr'] is None:
                g['lr'] = g['base_lr']

    # 2) warmup 리스트 자동 생성(필요 시)
    if getattr(c, 'lr_warm', False):
        if not hasattr(c, 'lr_warmup_from_list'):
            c.lr_warmup_from_list = [blr / 10.0 for blr in base_lrs]

        if not hasattr(c, 'lr_warmup_to_list'):
            if getattr(c, 'lr_cosine', False) and total_epoch is not None:
                # 워밍업 종료 시점(=c.lr_warm_epochs)에서 cosine의 상대 위치로 to값을 맞춰줌
                decay_pow = c.lr_decay_rate ** 3  # eta_min 비율
                cos_factor = (1 + math.cos(math.pi * c.lr_warm_epochs / float(total_epoch))) / 2.0
                c.lr_warmup_to_list = [
                    (blr * decay_pow) + (blr - blr * decay_pow) * cos_factor
                    for blr in base_lrs
                ]
            else:
                c.lr_warmup_to_list = base_lrs[:]  # cosine이 아니면 to=base_lr

        # 길이 일치 보장
        assert len(c.lr_warmup_from_list) == num
        assert len(c.lr_warmup_to_list)   == num

    return base_lrs


def warmup_group_learning_rate(c, epoch, batch_id, total_batches, optimizers, total_epoch=None):
    """
    배치 단위 워밍업 (NF별 optimizer 분리 버전).
    - 워밍업 구간(epoch < c.lr_warm_epochs)에서만 선형 증가.
    - 각 optimizer의 모든 param_group에 동일한 lr을 적용.
    - 반환: 각 optimizer(=NF)별 현재 lr 리스트 (길이=len(optimizers))
    """
    _ensure_base_lrs_and_warmup_lists_for_optims(c, optimizers, total_epoch)

    # 워밍업 비활성 또는 워밍업 구간 종료 시: 현재 lr들 그대로 반환
    if not getattr(c, "lr_warm", False) or epoch >= c.lr_warm_epochs:
        return [opt.param_groups[0]["lr"] for opt in optimizers]

    # 선형 워밍업 진행도 p 계산 (에폭×배치 기반)
    denom = max(1, int(c.lr_warm_epochs) * max(1, int(total_batches)))
    p = (int(batch_id) + int(epoch) * max(1, int(total_batches))) / float(denom)
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

    new_lrs = []
    for i, opt in enumerate(optimizers):
        lr_from = float(c.lr_warmup_from_list[i])
        lr_to   = float(c.lr_warmup_to_list[i])
        lr_i = lr_from + p * (lr_to - lr_from)

        for g in opt.param_groups:
            g["lr"] = lr_i
        new_lrs.append(lr_i)

    return new_lrs


def adjust_group_learning_rate(c, optimizers, epoch, total_epoch):
    """
    에폭 단위 스케줄 (NF별 optimizer 분리 버전).
    - 워밍업 에폭 동안은 배치 단위 워밍업이 조절하므로 여기선 건드리지 않음.
    - 워밍업 구간이 아닐 경우, cosine 또는 multistep로 각 optimizer의 lr를 갱신.
    - 반환: 각 optimizer별 현재 lr 리스트
    """
    # 워밍업 중이면 건드리지 않고 현 상태 반환
    if getattr(c, "lr_warm", False) and epoch < c.lr_warm_epochs:
        return [opt.param_groups[0]["lr"] for opt in optimizers]

    base_lrs = _ensure_base_lrs_and_warmup_lists_for_optims(c, optimizers, total_epoch)

    T = max(1, int(total_epoch))
    new_lrs = []

    if getattr(c, "lr_cosine", False):
        # Cosine annealing: eta_min = base_lr * (c.lr_decay_rate ** 3)
        r = float(c.lr_decay_rate) ** 3
        cos_factor = (1.0 + math.cos(math.pi * float(epoch) / float(T))) / 2.0
        for i, opt in enumerate(optimizers):
            lr_i = base_lrs[i] * (r + (1.0 - r) * cos_factor)
            for g in opt.param_groups:
                g["lr"] = lr_i
            new_lrs.append(lr_i)
    else:
        # Multi-step decay
        steps = 0
        if hasattr(c, "lr_decay_epochs"):
            steps = sum(int(epoch) >= int(m) for m in c.lr_decay_epochs)
        decay = (float(c.lr_decay_rate) ** int(steps)) if steps > 0 else 1.0

        for i, opt in enumerate(optimizers):
            lr_i = base_lrs[i] * decay
            for g in opt.param_groups:
                g["lr"] = lr_i
            new_lrs.append(lr_i)

    return new_lrs