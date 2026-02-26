""" Visformer

Paper: Visformer: The Vision-friendly Transformer - https://arxiv.org/abs/2104.12533

From original at https://github.com/danczs/Visformer

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""

import torch
import torch.nn as nn

from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import to_2tuple, trunc_normal_, DropPath, PatchEmbed, LayerNorm2d, create_classifier, use_fused_attn
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs

__all__ = ['VisformerDec']


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        groups = 1,
        norm_layer = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = conv3x3(inplanes, planes, stride)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out



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


class VisformerDec(nn.Module):
    def __init__(
            self,
            uplayers = [1,2,3],
            skip_layers = [1,2,3],
            skip_connection = False,
            num_classes=3,
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225],
            is_pairset = True,
            final_activation = 'sigmoid',
            patch_size=16,
            in_channels=384,
            conv_init_channels = 16,
            channel_reduction = 8,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=LayerNorm2d,
            attn_stage='111',
            spatial_conv='111',
            group=8,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.in_channels = in_channels
        self.mean = mean 
        self.std = std

        self.conv_init_channels = conv_init_channels
        self.skip_connection = skip_connection
        self.skip_layers = skip_layers
        self.up_layers_channel_init = self.in_channels//channel_reduction
        self.upsample_layer = nn.Upsample(scale_factor = 2, mode ="bilinear", align_corners = True)
        self.uplayers_list = nn.ModuleList()
        for l in range(max(self.skip_layers)-1):
            dec_l = max(self.skip_layers)-l-1
            if l==0:
                if (dec_l in self.skip_layers)==True and self.skip_connection==True:
                    if is_pairset ==True:
                        inplanes = self.in_channels+self.in_channels//4
                    else:
                        inplanes = self.in_channels+self.in_channels//2
                else:
                    inplanes = self.in_channels
            else:
                if (dec_l in self.skip_layers)==True and self.skip_connection==True:
                    if is_pairset ==True:
                        inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+2))
                    else:
                        inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+1))
                else:
                    inplanes = self.up_layers_channel_init//(2**(l-1))
            planes = self.up_layers_channel_init//(2**l)
            stage = nn.Sequential(*[
                Block(
                    dim=inplanes,
                    num_heads=num_heads,
                    head_dim_ratio=1.0,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    group=group,
                    attn_disabled=(attn_stage[dec_l] == '0'),
                    spatial_conv=(spatial_conv[dec_l] == '1'),
                )
                for i in range(depth[dec_l])
            ] + [ConvBlock(inplanes, planes)])
            self.uplayers_list.append(stage)

        self.conv_block1 = ConvBlock(planes, self.conv_init_channels*2)
        self.conv_block2 = ConvBlock(self.conv_init_channels*2, self.conv_init_channels*2)

        if (0 in self.skip_layers)==True and self.skip_connection==True:
            self.conv_block3 = ConvBlock(self.conv_init_channels*3, self.conv_init_channels)
        else:
            self.conv_block3 = ConvBlock(self.conv_init_channels*2, self.conv_init_channels)
        self.last_conv = nn.Conv2d(self.conv_init_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        if final_activation =='sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation =='softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation =='relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif final_activation == 'leaky':
            self.final_activation = nn.LeakyReLU(inplace=True)
        else:
            self.final_activation = nn.Identity()

        self.normalize = T.Normalize(self.mean, self.std)
        self.num_classes = num_classes

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

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

    def forward(self, x):
        x_in = x
        x = x_in[-1]
        l_skip=-2

        for l in range(max(self.skip_layers)-1):
            dec_l = max(self.skip_layers)-l-1
            x = self.upsample_layer(x)
            if (dec_l in self.skip_layers)==True and self.skip_connection==True:
                x = torch.cat((x_in[l_skip], x), 1)
                l_skip -= 1
            else:
                pass
            x = self.uplayers_list[l](x)

        x = self.upsample_layer(x)
        x = self.conv_block1(x)
        x = self.upsample_layer(x)
        x = self.conv_block2(x)
        if (0 in self.skip_layers)==True and self.skip_connection==True:
            x = torch.cat((x_in[l_skip], x), 1)
        else:
            pass
        x = self.upsample_layer(x)
        x = self.conv_block3(x)

        x = self.last_conv(x)
        x = self.final_activation(x)
        if self.num_classes==3:
            x = self.normalize(x)
        return x


@register_model
def visformer_tiny_dec(num_classes, skip_layers, final_activation, skip_connection, in_channels, channel_reduction, is_pairset):
    model_cfg = dict(
        conv_init_channels=16, depth=(7, 4, 4), num_heads=3, mlp_ratio=4., group=8,
        attn_stage='011', spatial_conv='100', norm_layer=nn.BatchNorm2d, 
        num_classes = num_classes, skip_layers = skip_layers, final_activation = final_activation, skip_connection = skip_connection, in_channels = in_channels,  channel_reduction = channel_reduction, is_pairset = is_pairset)
    model = VisformerDec(**model_cfg)
    return model




