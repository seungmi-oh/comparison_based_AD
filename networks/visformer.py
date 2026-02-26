""" Visformer

Paper: Visformer: The Vision-friendly Transformer - https://arxiv.org/abs/2104.12533

From original at https://github.com/danczs/Visformer

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import to_2tuple, trunc_normal_, DropPath, PatchEmbed, LayerNorm2d, create_classifier, use_fused_attn
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs

from .attention import *

__all__ = ['Visformer', 'Twin_Visformer', 'visformer_tiny']


class SpatialMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.,
            group=8,
            spatial_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2:  # net setting
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, groups=self.group, bias=False)
            self.act2 = act_layer()
        else:
            self.conv2 = None
            self.act2 = None
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)
        self.drop3 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)
        if self.conv2 is not None:
            x = self.conv2(x)
            x = self.act2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        return x


class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, head_dim_ratio=1., attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        q, k, v = x.unbind(0)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            head_dim_ratio=1.,
            mlp_ratio=4.,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
            group=8,
            attn_disabled=False,
            spatial_conv=False,
    ):
        super().__init__()
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if attn_disabled:
            self.norm1 = None
            self.attn = None
        else:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                head_dim_ratio=head_dim_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )

        self.norm2 = norm_layer(dim)
        self.mlp = SpatialMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            group=group,
            spatial_conv=spatial_conv,
        )

    def forward(self, x):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Visformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            init_channels=32,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=LayerNorm2d,
            attn_stage='111',
            use_pos_embed=True,
            spatial_conv='111',
            vit_stem=False,
            group=8,
            global_pool='avg',
            conv_init=False,
            embed_norm=None,
            out_layers = [1,2,3],
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_stem = vit_stem
        self.conv_init = conv_init
        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.use_pos_embed = use_pos_embed
        self.grad_checkpointing = False

        self.out_layers = out_layers
        self.feats = [torch.zeros(1) for i in range(len(self.out_layers))]
        self.nf_skips = [nn.Identity(name=f'nf_skip{i}') for i in range(len(self.out_layers))]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # stage 1
        if self.vit_stem:
            self.stem = None
            self.patch_embed1 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // patch_size for x in img_size]
        else:
            if self.init_channels is None:
                self.stem = None
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size // 2,
                    in_chans=in_chans,
                    embed_dim=embed_dim // 2,
                    norm_layer=embed_norm,
                    flatten=False,
                )
                img_size = [x // (patch_size // 2) for x in img_size]
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, self.init_channels, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.init_channels),
                    nn.ReLU(inplace=True)
                )
                img_size = [x // 2 for x in img_size]
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size // 4,
                    in_chans=self.init_channels,
                    embed_dim=embed_dim // 2,
                    norm_layer=embed_norm,
                    flatten=False,
                )
                img_size = [x // (patch_size // 4) for x in img_size]

        if self.use_pos_embed:
            if self.vit_stem:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, *img_size))
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
        else:
            self.pos_embed1 = None

        self.stage1 = nn.Sequential(*[
            Block(
                dim=embed_dim//2,
                num_heads=num_heads,
                head_dim_ratio=0.5,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[0] == '0'),
                spatial_conv=(spatial_conv[0] == '1'),
            )
            for i in range(self.stage_num1)
        ])

        # stage2
        if not self.vit_stem:
            self.patch_embed2 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size // 8,
                in_chans=embed_dim // 2,
                embed_dim=embed_dim,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // (patch_size // 8) for x in img_size]
            if self.use_pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed2 = None
        else:
            self.patch_embed2 = None
        self.stage2 = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[1] == '0'),
                spatial_conv=(spatial_conv[1] == '1'),
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        if not self.vit_stem:
            self.patch_embed3 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size // 8,
                in_chans=embed_dim,
                embed_dim=embed_dim * 2,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // (patch_size // 8) for x in img_size]
            if self.use_pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, *img_size))
            else:
                self.pos_embed3 = None
        else:
            self.patch_embed3 = None
        self.stage3 = nn.Sequential(*[
            Block(
                dim=embed_dim * 2,
                num_heads=num_heads,
                head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[2] == '0'),
                spatial_conv=(spatial_conv[2] == '1'),
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        self.num_features = embed_dim if self.vit_stem else embed_dim * 2
        self.norm = norm_layer(self.num_features)

        # head
        global_pool, head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(drop_rate)
        self.head = head

        # weights init
        if self.use_pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_stem:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def freeze_bn_layers(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                param.requires_grad = False

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed1|pos_embed1|stem',  # stem and embed
            blocks=[
                (r'^stage(\d+)\.(\d+)' if coarse else r'^stage(\d+)\.(\d+)', None),
                (r'^(?:patch_embed|pos_embed)(\d+)', (0,)),
                (r'^norm', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        l=0
        outputs = []
        if self.stem is not None:
            x = self.stem(x)

        if 0 in self.out_layers:
            x = self.nf_skips[l](x)
            outputs.append(x)
            self.feats[l] = x
            l+=1

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed1 is not None:
            x = self.pos_drop(x + self.pos_embed1)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage1, x)
        else:
            x = self.stage1(x)

        if max(self.out_layers)>0:
            x = self.nf_skips[l](x)
            outputs.append(x)
            self.feats[l] = x
            l+=1

        # stage 2
        if self.patch_embed2 is not None:
            x = self.patch_embed2(x)
            if self.pos_embed2 is not None:
                x = self.pos_drop(x + self.pos_embed2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage2, x)
        else:
            x = self.stage2(x)

        if max(self.out_layers)>1:
            x = self.nf_skips[l](x)
            outputs.append(x)
            self.feats[l] = x
            l+=1

        # stage3
        if self.patch_embed3 is not None:
            x = self.patch_embed3(x)
            if self.pos_embed3 is not None:
                x = self.pos_drop(x + self.pos_embed3)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage3, x)
        else:
            x = self.stage3(x)

        if max(self.out_layers)>2:
            x = self.nf_skips[l](x)
            outputs.append(x)
            self.feats[l] = x
            l+=1

        x = self.norm(x)
        return outputs

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
#        x = self.forward_head(x)
        return x


class Twin_Visformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            init_channels=32,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=LayerNorm2d,
            attn_stage='111',
            use_pos_embed=True,
            spatial_conv='111',
            vit_stem=False,
            group=8,
            global_pool='avg',
            conv_init=False,
            embed_norm=None,
            out_layers = [1,2,3],
            finetuning = True,
            att_layers = [1,2],
            att_type = '',
            concat_last_feat=True,
            nf_inp = ['diff']
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_stem = vit_stem
        self.conv_init = conv_init
        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.use_pos_embed = use_pos_embed
        self.grad_checkpointing = False

        self.concat_last_feat = concat_last_feat
        self.out_layers = out_layers
        self.feats = [torch.zeros(1) for i in range(len(self.out_layers))]
        self.nf_skips = [nn.Identity(name=f'nf_skip{i}') for i in range(len(self.out_layers))]
        self.dec_skips = [nn.Identity(name=f'dec_skip{i}') for i in range(len(self.out_layers))]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # stage 1
        if self.vit_stem:
            self.stem = None
            self.patch_embed1 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // patch_size for x in img_size]
        else:
            if self.init_channels is None:
                self.stem = None
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size // 2,
                    in_chans=in_chans,
                    embed_dim=embed_dim // 2,
                    norm_layer=embed_norm,
                    flatten=False,
                )
                img_size = [x // (patch_size // 2) for x in img_size]
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, self.init_channels, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.init_channels),
                    nn.ReLU(inplace=True)
                )
                img_size = [x // 2 for x in img_size]
                self.patch_embed1 = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size // 4,
                    in_chans=self.init_channels,
                    embed_dim=embed_dim // 2,
                    norm_layer=embed_norm,
                    flatten=False,
                )
                img_size = [x // (patch_size // 4) for x in img_size]

        if self.use_pos_embed:
            if self.vit_stem:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, *img_size))
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
        else:
            self.pos_embed1 = None

        self.stage1 = nn.Sequential(*[
            Block(
                dim=embed_dim//2,
                num_heads=num_heads,
                head_dim_ratio=0.5,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[0] == '0'),
                spatial_conv=(spatial_conv[0] == '1'),
            )
            for i in range(self.stage_num1)
        ])

        # stage2
        if not self.vit_stem:
            self.patch_embed2 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size // 8,
                in_chans=embed_dim // 2,
                embed_dim=embed_dim,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // (patch_size // 8) for x in img_size]
            if self.use_pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed2 = None
        else:
            self.patch_embed2 = None
        self.stage2 = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[1] == '0'),
                spatial_conv=(spatial_conv[1] == '1'),
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        if not self.vit_stem:
            self.patch_embed3 = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size // 8,
                in_chans=embed_dim,
                embed_dim=embed_dim * 2,
                norm_layer=embed_norm,
                flatten=False,
            )
            img_size = [x // (patch_size // 8) for x in img_size]
            if self.use_pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, *img_size))
            else:
                self.pos_embed3 = None
        else:
            self.patch_embed3 = None
        self.stage3 = nn.Sequential(*[
            Block(
                dim=embed_dim * 2,
                num_heads=num_heads,
                head_dim_ratio=1.0,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                group=group,
                attn_disabled=(attn_stage[2] == '0'),
                spatial_conv=(spatial_conv[2] == '1'),
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        self.num_features = embed_dim if self.vit_stem else embed_dim * 2
        self.norm = norm_layer(self.num_features)

        # head
        global_pool, head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(drop_rate)
        self.head = head

        # weights init
        if self.use_pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_stem:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

        # set co-attention modules 
        self.finetuning = finetuning
        self.att_layers = att_layers
        self.att_type = att_type
        self.coattention_modules = nn.ModuleList()
        if len(self.att_type)>0 and self.finetuning==True:
            for l_att in self.att_layers:
                if l_att ==0:
                    in_channels = self.init_channels
                else:
                    in_channels = embed_dim * (2 ** (l_att-2)) 

                if 'SpCo' == self.att_type:
                    self.coattention_modules.append(SpCoAtt(in_channels, reduction = 1))
                elif 'Comb' == self.att_type:
                    self.coattention_modules.append(CombAtt(in_channels, reduction = 1))
                elif 'CoCo' == self.att_type:
                    self.coattention_modules.append(CoCoAtt(in_channels, reduction = 1))
                else:
                    raise KeyboardInterrupt
        else:
            pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def freeze_bn_layers(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                param.requires_grad = False

    @torch.jit.ignore
    def freeze_all_layers(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def unfreeze_all_layers(self):
        for name, param in self.named_parameters():
            param.requires_grad = True 

    @torch.jit.ignore
    def unfreeze_att_conv_layers(self):
        for name, param in self.coattention_modules.named_parameters():
            param.requires_grad = True 

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed1|pos_embed1|stem',  # stem and embed
            blocks=[
                (r'^stage(\d+)\.(\d+)' if coarse else r'^stage(\d+)\.(\d+)', None),
                (r'^(?:patch_embed|pos_embed)(\d+)', (0,)),
                (r'^norm', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_once(self, x):
        l=0
        outputs = []
        if self.stem is not None:
            x = self.stem(x)

        if 0 in self.out_layers:
            outputs.append(x)
            l+=1

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed1 is not None:
            x = self.pos_drop(x + self.pos_embed1)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage1, x)
        else:
            x = self.stage1(x)

        if max(self.out_layers)>0:
            outputs.append(x)
            l+=1

        # stage 2
        if self.patch_embed2 is not None:
            x = self.patch_embed2(x)
            if self.pos_embed2 is not None:
                x = self.pos_drop(x + self.pos_embed2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage2, x)
        else:
            x = self.stage2(x)

        if max(self.out_layers)>1:
            outputs.append(x)
            l+=1

        # stage3
        if self.patch_embed3 is not None:
            x = self.patch_embed3(x)
            if self.pos_embed3 is not None:
                x = self.pos_drop(x + self.pos_embed3)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage3, x)
        else:
            x = self.stage3(x)

        if max(self.out_layers)>2:
            outputs.append(x)
            l+=1

        x = self.norm(x)
        return outputs

    def forward_features(self, x1, x2):
        # See note [TorchScript super()]
        outputs = []
        x1_list = self.forward_once(x1)
        x2_list = self.forward_once(x2)

        for i in range(len(self.out_layers)):
            x1_feat = x1_list[i] 
            x2_feat = x2_list[i] 
            l = self.out_layers[i]
            if l in self.att_layers: 
                a_idx = self.att_layers.index(l)
                # 2d-Flow features of a layer with attention modules for the finetuned feature extractor 
                if self.finetuning==True:
                    if 'SpCo' == self.att_type:
                        x21_feat, x12_feat, transformed_x1, transformed_x2, self.transformed = self.coattention_modules[a_idx](x1_feat, x2_feat)
                    elif 'Comb' == self.att_type:
                        x21_feat, x12_feat, transformed_x1, transformed_x2, self.transformed = self.coattention_modules[a_idx](x1_feat, x2_feat)
                    elif 'CoCo' == self.att_type:
                        x21_feat, x12_feat, transformed_x1, transformed_x2, self.transformed = self.coattention_modules[a_idx](x1_feat, x2_feat) 
                    else:
                        raise KeyboardInterrupt
                    self.diff = torch.abs(x1_feat-x2_feat) 
                    self.att_diff = torch.abs(x21_feat-x12_feat)
                    self.transformed_diff = torch.abs(transformed_x1-transformed_x2)

                    x = self.dec_skips[i](self.transformed)
                    outputs.append(x)

                    feature_list = [0.0 for f_num in range(len(self.nf_inp))]
                    for f_num in range(len(self.nf_inp)):
                        feat_name = self.nf_inp[f_num]
                        feature_list[f_num] = eval(f'self.{feat_name}')

                    if len(self.nf_inp)>1:
                        x_nf = self.nf_skips[i](torch.cat(feature_list, dim=1))
                    else:
                        x_nf = self.nf_skips[i](feature_list[0])

                    self.feats[i] = [x1_feat, x2_feat, self.diff, x21_feat, x12_feat, self.att_diff, transformed_x1, transformed_x2, self.transformed_diff, self.transformed]

                # 2d-Flow features of a layer with attention modules for the pretrained feature extractor 
                else:
                    if 'SpCo' == self.att_type:
                        x12_feat, x21_feat = spatial_coattention(x1_feat, x2_feat)
                    elif 'ChCo' ==self.att_type:
                        x12_feat, x21_feat = channels_coattention(x1_feat, x2_feat)
                    elif 'Comb' == self.att_type:
                        x12_feat, x21_feat = combined_coattention(x1_feat, x2_feat)  
                    else:
                        raise KeyboardInterrupt
                    self.diff = torch.abs(x1_feat-x2_feat) 
                    transformed_x1 = torch.cat((x1_feat, x21_feat), dim=1)
                    transformed_x2 = torch.cat((x2_feat, x12_feat), dim=1)
                    self.transformed_diff = torch.abs(transformed_x1-transformed_x2)

                    feature_list = [0.0 for f_num in range(len(self.nf_inp))]
                    for f_num in range(len(self.nf_inp)):
                        feat_name = self.nf_inp[f_num]
                        feature_list[f_num] = eval(f'self.{feat_name}')

                    if len(self.nf_inp)>1:
                        x_nf = self.nf_skips[i](torch.cat(feature_list, dim=1))
                    else:
                        x_nf = self.nf_skips[i](feature_list[0])

                    self.feats[i] = [x1_feat, x2_feat, self.diff, x21_feat, x12_feat, torch.abs(x21_feat-x12_feat), transformed_x1, transformed_x2, self.transformed_diff]

            # 2d-Flow features of a layer without attention modules for a feature extractor 
            else:
                diff = torch.abs(x1_feat-x2_feat)
                x_nf = self.nf_skips[i](diff)
                self.feats[i] = [x1_feat, x2_feat, diff]

                if i== len(self.out_layers)-1 and self.concat_last_feat ==True:
                    x = self.dec_skips[i](torch.cat((x1_feat,x2_feat), dim=1))
                else:
                    x = self.dec_skips[i](torch.abs(x1_feat-x2_feat))
                outputs.append(x)
        return outputs 

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)
#        x = self.forward_head(x)
        return x


def _create_visformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(Visformer, variant, pretrained, **kwargs)
    return model


def _create_twin_visformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(Twin_Visformer, variant, pretrained, **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'visformer_tiny.in1k': _cfg(hf_hub_id='timm/'),
    'visformer_small.in1k': _cfg(hf_hub_id='timm/'),
})


@register_model
def visformer_tiny(is_pairset, pool_layers, att_layers, att_type, pretrained, finetuning, concat_last_feat, nf_inp, **kwargs):
    if is_pairset == True:
        model_cfg = dict(
            init_channels=16, embed_dim=192, depth=(7, 4, 4), num_heads=3, mlp_ratio=4., group=8,
            attn_stage='011', spatial_conv='100', norm_layer=nn.BatchNorm2d, conv_init=True,
            embed_norm=nn.BatchNorm2d, out_layers=pool_layers, att_layers = att_layers, att_type = att_type, finetuning = finetuning, concat_last_feat = concat_last_feat, nf_inp = nf_inp , **kwargs)
        model = _create_twin_visformer('visformer_tiny.in1k', pretrained=pretrained, **model_cfg)
    else:
        model_cfg = dict(
            init_channels=16, embed_dim=192, depth=(7, 4, 4), num_heads=3, mlp_ratio=4., group=8,
            attn_stage='011', spatial_conv='100', norm_layer=nn.BatchNorm2d, conv_init=True,
            embed_norm=nn.BatchNorm2d, out_layers=pool_layers, **kwargs)
        model = _create_visformer('visformer_tiny.in1k', pretrained=pretrained, **model_cfg)
    return model


