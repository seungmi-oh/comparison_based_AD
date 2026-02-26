import torch
import torch.nn as nn
import torch.nn.functional as F

def match_scale_global(x, ref, eps=1e-8):
    """
    x를 ref의 전역 평균(mean over B,C,H,W가 아니라 B축 제외: C,H,W)에 맞춤.
    x, ref: (B,C,H,W)
    반환: x * (mean(ref)/mean(x))
    """
    mx  = x.mean(dim=(1,2,3), keepdim=True)     # (B,1,1,1)
    mref= ref.mean(dim=(1,2,3), keepdim=True)   # (B,1,1,1)
    scale = mref / (mx + eps)
    return x * scale

def match_scale_channel_mean(x, ref, eps=1e-8):
    """
    x를 ref의 채널별 공간 평균(mean over H,W per channel)에 맞춤.
    x, ref: (B,C,H,W)
    반환: x * (mean_HW(ref)/mean_HW(x)), 채널별 스케일
    """
    mx  = x.mean(dim=(-2,-1), keepdim=True)     # (B,C,1,1)
    mref= ref.mean(dim=(-2,-1), keepdim=True)   # (B,C,1,1)
    scale = mref / (mx + eps)
    return x * scale

# channel co-attention without trainable parameters
def channel_coattention(Fa, Fb, one_sided=False, normalized = 'none'):
    B,C,H,W = Fa.size()
    Fa_r = torch.reshape(Fa, (B,C,H*W)) # BxCaxM
    Fb_r = torch.reshape(Fb, (B,C,H*W)) # BxCbxM
    Fa_r_tr = torch.permute(Fa_r, (0,2,1)) #BxMxCa
    Fb_r_tr = torch.permute(Fb_r, (0,2,1)) #BxMxCb
    ChCo = F.cosine_similarity(Fa_r.unsqueeze(2), Fb_r.unsqueeze(1), dim=-1) #BxCaxCb
    if normalized=='sum':
        ChCo_Fa = ChCo/(torch.sum(ChCo, dim=-2, keepdims=True)+1e-8)
    elif normalized == 'num':
        ChCo_Fa = ChCo/(ChCo.size(-2)+1e-8)
    elif normalized == 'both':
        ChCo_Fa = ChCo/(ChCo.size(-2)*torch.sum(ChCo, dim=-2, keepdims=True)+1e-8)
    elif normalized == 'none':
        ChCo_Fa = ChCo
    else:
        pass
    Fab = torch.matmul(Fa_r_tr, ChCo_Fa) # BxMxCb 
    Fab = torch.reshape(torch.permute(Fab, (0,2,1)), (B,C,H,W)) 
    if one_sided==False:
        if normalized=='sum':
            ChCo_Fb = ChCo/(torch.sum(ChCo, dim=-1, keepdims=True)+1e-8)
        elif normalized == 'num':
            ChCo_Fb = ChCo/(ChCo.size(-1)+1e-8)
        elif normalized == 'both':
            ChCo_Fb = ChCo/(ChCo.size(-1)*torch.sum(ChCo, dim=-1, keepdims=True)+1e-8)
        elif normalized == 'none':
            ChCo_Fb = ChCo
        else:
            pass
        Fba = torch.matmul(Fb_r_tr, torch.permute(ChCo_Fb, (0,2,1))) #BxMxCa 
        Fba = torch.reshape(torch.permute(Fba, (0,2,1)), (B,C,H,W)) 
        return Fab, Fba
    else:
        return Fab

# spatial co-attention without trainable parameters
def spatial_coattention(Fa,Fb, normalized='num'):
    B,C,H,W = Fa.size()
    M = H*W
    Fa_r = torch.reshape(Fa, (B,C,H*W)) # BxCxMa
    Fa_r_tr = torch.permute(Fa_r, (0,2,1)) #BxMaxC
    Fb_r = torch.reshape(Fb, (B,C,H*W)) # BxCxMb
    Fb_r_tr = torch.permute(Fb_r, (0,2,1)) #BxMbxC
    SpCo = torch.zeros((B,M,M))
    SpCo = SpCo.to(Fa.device)
    Fab = torch.zeros_like(Fa_r)
    Fba = torch.zeros_like(Fb_r)
    # due to GPU memory limitations 
    for i in range(B):
        SpCo[i,:,:] =nxn_cos_sim(Fa_r_tr[i,:,:], Fb_r_tr[i,:,:]) 

    if normalized=='sum':
        SpCo_Fa = SpCo/(torch.sum(SpCo, dim=-2, keepdims=True)+1e-8)
    elif normalized == 'num':
        SpCo_Fa = SpCo/(SpCo.size(-2)+1e-8)
    elif normalized == 'both':
        SpCo_Fa = SpCo/(SpCo.size(-2)*torch.sum(SpCo, dim=-2, keepdims=True)+1e-8)
    elif normalized == 'none':
        SpCo_Fa = SpCo
    else:
        pass
    Fab = torch.matmul(Fa_r, SpCo_Fa) # BxCxMb 

    if normalized=='sum':
        SpCo_Fb = SpCo/(torch.sum(SpCo, dim=-1, keepdims=True)+1e-8)
    elif normalized == 'num':
        SpCo_Fb = SpCo/(SpCo.size(-1)+1e-8)
    elif normalized == 'both':
        SpCo_Fb = SpCo/(SpCo.size(-1)*torch.sum(SpCo, dim=-1, keepdims=True)+1e-8)
    elif normalized == 'none':
        SpCo_Fb = SpCo
    else:
        pass

    Fba = torch.matmul(Fb_r, torch.permute(SpCo_Fb, (0,2,1))) #BxCxMa 
    Fab = torch.reshape(Fab, (B,C,H,W))  
    Fba = torch.reshape(Fba,(B,C,H,W))
    return Fab, Fba

# get cosine similairy matrix
def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)


# combined co-attention without trainable parameters
def combined_coattention(Fa, Fb, normalized ='none', search_half_range=1, padding_mode = 'replicate'):
    Fa_pad = torch.nn.functional.pad(Fa, [search_half_range, search_half_range, search_half_range, search_half_range], padding_mode)
    Fb_pad = torch.nn.functional.pad(Fb, [search_half_range, search_half_range, search_half_range, search_half_range], padding_mode)

    search_range = 2 * search_half_range + 1
    feat_h, feat_w = Fa.size(-2), Fb.size(-1)
    Fab = torch.zeros_like(Fa)
    Fba = torch.zeros_like(Fb)
    for s in range(search_range ** 2):
        y = s // search_range
        x = s % search_range

        Fb_shift = Fb_pad[:,:,y:y+feat_h, x:x+feat_w] 
        Fba_shift = channel_coattention(Fb_shift, Fa, one_sided=True, normalized = normalized) #Fb_shift 를 Fa 처럼 변형
        Fba_shift_Fa_similarity = get_similarity(Fba_shift, Fa) # Fba_shift 랑 Fa 의 유사도 구함
        Fba += torch.mul(Fba_shift_Fa_similarity, Fba_shift)  # Fba_shift 에 Fa 와의 유사도 반영해서 최종 변형 결과 생성

        Fa_shift = Fa_pad[:,:,y:y+feat_h, x:x+feat_w] 
        Fab_shift = channel_coattention(Fa_shift, Fb, one_sided=True, normalized = normalized)
        Fab_shift_Fb_similarity = get_similarity(Fab_shift, Fb)
        Fab += torch.mul(Fab_shift_Fb_similarity, Fab_shift) 

    Fab = torch.div(torch.mul(Fab,torch.mean(Fb, dim=(-2,-1), keepdims=True)),torch.mean(Fab, dim=(-2,-1), keepdims=True)+1e-8) # Fb scale 로 맞추는 과정)
    Fba = torch.div(torch.mul(Fba,torch.mean(Fa, dim=(-2,-1), keepdims=True)),torch.mean(Fba, dim=(-2,-1), keepdims=True)+1e-8) 
    return Fab, Fba


def get_similarity(x1, x2):
    x1_norm = torch.linalg.norm(x1.view([-1,x1.size(1),x1.size(2)*x1.size(3)]), dim=(1,2), keepdims=True)
    x1_norm = torch.unsqueeze(x1_norm, -1) 
    x2_norm = torch.linalg.norm(x2.view([-1,x2.size(1),x2.size(2)*x2.size(3)]), dim=(1,2), keepdims=True)
    x2_norm = torch.unsqueeze(x2_norm, -1) 
    similarity = torch.sum(torch.div(torch.mul(x1, x2),torch.mul(x1_norm, x2_norm)+1e-8), dim=(1,2,3), keepdims=True)
    return similarity

def get_similarity_1d(r1: torch.Tensor, r2: torch.Tensor):
    """
    r1, r2: (B,C,L) 
    return: similarity (B,)  - 전체 L 위치 평균 cosine similarity
    """
    B, C, L = r1.shape
    # flatten (B, C*L)
    r1_flat = r1.reshape(B, -1)
    r2_flat = r2.reshape(B, -1)

    # norm
    r1_norm = torch.norm(r1_flat, p=2, dim=1, keepdim=True) + 1e-8
    r2_norm = torch.norm(r2_flat, p=2, dim=1, keepdim=True) + 1e-8

    # cosine similarity
    sim = (r1_flat * r2_flat).sum(dim=1, keepdim=True) / (r1_norm * r2_norm)
    return sim.squeeze(1)   # (B,)

class CoCoAtt(nn.Module):
    def __init__(self, inp: int,
                 temp: float = 0.05, max_disp: int = 3, padding_mode: str = "replicate", align_corners: bool = True):
        super().__init__()
        assert inp > 0
        self.C = inp
        self.temp = float(temp)
        self.D = int(max_disp)
        # self.gate_eps = float(gate_eps)
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        self.conv = torch.nn.Conv2d(2*inp, inp, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True) 
        
    def _corr1d_expectation_rigid(self, r1, r2, temp: float):
        B, C, L = r1.shape

        scores = []  # 각 d에 대한 전역 스코어 (B,)
        Ds = range(-self.D, self.D + 1)
        for d in Ds:
            if d >= 0:
                a = r1[:, :, d:]           # (B,C,L-d)
                b = r2[:, :, :L - d]       # (B,C,L-d)
            else:
                d2 = -d
                a = r1[:, :, :L - d2]      # (B,C,L-d2)
                b = r2[:, :, d2:]          # (B,C,L-d2)
            sim = get_similarity_1d(a, b)  # (B,)
            scores.append(sim)    

        prob = torch.stack(scores, dim=1)         # (B, 2D+1)
        prob = F.softmax(prob / temp, dim=1)      # (B, 2D+1)

        disp_vals = torch.arange(-self.D, self.D + 1, device=r1.device, dtype=r1.dtype)  # (2D+1,)
        mu = (prob * disp_vals[None, :]).sum(dim=1)    # (B,)
        return mu, prob

    def _make_joint_outer(self, prob_h, prob_w):
        # prob_h: (B,2D+1), prob_w: (B,2D+1)
        # prob_hw: (B,2D+1,2D+1)  with sum=1 per batch
        prob_hw = prob_h.unsqueeze(2) * prob_w.unsqueeze(1)  # outer product
        prob_hw = prob_hw / (prob_hw.sum(dim=(1,2), keepdim=True) + 1e-8)
        return prob_hw

    def _shift_pad(self, x, d, dim, pad_mode="replicate"):
        # x:(B,C,H,W), dim=2(H) or 3(W), d: int shift
        B,C,H,W = x.shape
        if d == 0:
            return x  # 이동 없음이면 패딩 호출 자체를 생략해 에러 방지

        B, C, H, W = x.shape
        mode = pad_mode

        # reflect는 pad>0 & pad < size 제약 → 범위 클램프
        if mode == 'reflect':
            if dim == 2:
                d = max(min(abs(d), H-1), 1) * (1 if d > 0 else -1)
            else:
                d = max(min(abs(d), W-1), 1) * (1 if d > 0 else -1)

        if dim == 2:  # 세로 이동
            if d > 0:
                pad = (0, 0, d, 0)          # 위쪽 d만큼 복제/반사 패딩
                xpad = F.pad(x, pad, mode=mode)
                return xpad[:, :, :H, :]
            else:
                d2 = -d
                pad = (0, 0, 0, d2)         # 아래쪽 d2 패딩
                xpad = F.pad(x, pad, mode=mode)
                return xpad[:, :, d2:, :]

        else:        # 가로 이동
            if d > 0:
                pad = (d, 0, 0, 0)          # 왼쪽 d 패딩
                xpad = F.pad(x, pad, mode=mode)
                return xpad[:, :, :, :W]
            else:
                d2 = -d
                pad = (0, d2, 0, 0)         # 오른쪽 d2 패딩
                xpad = F.pad(x, pad, mode=mode)
                return xpad[:, :, :, d2:]

    def _apply_weighted_shifts_2d(self, x, prob_hw, D, pad_mode="replicate"):
        B = x.size(0)
        out = 0.0
        ds = range(-D, D+1)
        for i, dh in enumerate(ds):
            for j, dw in enumerate(ds):
                w   = prob_hw[:, i, j].view(B,1,1,1)           # (B,1,1,1)
                x_d = self._shift_pad(self._shift_pad(x, dh, 2, pad_mode), dw, 3, pad_mode)
                out = out + w * x_d
        return out

    # ---------- 메인 ----------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape
        diff = torch.abs(x1-x2) 
          
        # channel co-attention 추가
        x21_ch = channel_coattention(x2, x1, one_sided=True)
        x12_ch = channel_coattention(x1, x2, one_sided=True)

        # 1) 축별 1D 프로필 
        x1_h = x1.mean(dim=3)  # (B,C,H)
        x2_h = x2.mean(dim=3)
        x1_w = x1.mean(dim=2)  # (B,C,W)
        x2_w = x2.mean(dim=2)
        x12_h = x12_ch.mean(dim=3)  # (B,C,H)
        x21_h = x21_ch.mean(dim=3)
        x12_w = x12_ch.mean(dim=2)  # (B,C,W)
        x21_w = x21_ch.mean(dim=2)

        # 2) cross-correlation → 기대변위 (H, W)
        mu_h_21, prob_h_21 = self._corr1d_expectation_rigid(x1_h, x21_h, self.temp)  # x2→x1의 세로 변위
        mu_w_21, prob_w_21 = self._corr1d_expectation_rigid(x1_w, x21_w, self.temp)  # x2→x1의 가로 변위
        mu_h_12, prob_h_12 = self._corr1d_expectation_rigid(x2_h, x12_h, self.temp)  # x1→x2
        mu_w_12, prob_w_12 = self._corr1d_expectation_rigid(x2_w, x12_w, self.temp)

        prob_hw_21 = self._make_joint_outer(prob_h_21, prob_w_21)
        prob_hw_12 = self._make_joint_outer(prob_h_12, prob_w_12)
        
        x21_feat = self._apply_weighted_shifts_2d(x21_ch, prob_hw_21, self.D, self.padding_mode)
        x12_feat = self._apply_weighted_shifts_2d(x12_ch, prob_hw_12, self.D, self.padding_mode)
        
        x21_feat = match_scale_channel_mean(x21_feat, x1)
        x12_feat = match_scale_channel_mean(x12_feat, x2)

        transformed_diff = torch.abs(x21_feat-x12_feat)

        transformed = self.relu(self.conv(torch.cat((diff, transformed_diff), dim=1)))
        return x21_feat, x12_feat, transformed


# spatial co-attention modules with trainable parameters
class SpCoAtt(torch.nn.Module):
    def __init__(self, inp, reduction=1):
        super(SpCoAtt, self).__init__()
        self.inp = inp
        self.conv = torch.nn.Conv2d(2*inp, inp, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True) 

    def forward(self, x1_feat, x2_feat):
        diff = torch.abs(x1_feat-x2_feat) 

        x12_feat, x21_feat = spatial_coattention(x1_feat, x2_feat)
        
        transformed_diff = torch.abs(x21_feat-x12_feat)

        transformed = self.relu(self.conv(torch.cat((diff, transformed_diff), dim=1)))
        return x21_feat, x12_feat, transformed


# combined co-attention modules with trainable parameters
class CombAtt(torch.nn.Module):
    def __init__(self, inp, reduction=1):
        super(CombAtt, self).__init__()
        self.inp = inp
        self.conv = torch.nn.Conv2d(2*inp, inp, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True) 

    def forward(self, x1_feat, x2_feat):
        diff = torch.abs(x1_feat-x2_feat) 

        x12_feat, x21_feat = combined_coattention(x1_feat, x2_feat)
        
        transformed_diff = torch.abs(x21_feat-x12_feat)

        transformed = self.relu(self.conv(torch.cat((diff, transformed_diff), dim=1)))
        return x21_feat, x12_feat, transformed
