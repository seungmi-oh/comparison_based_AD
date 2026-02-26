# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from typing import Callable, List, Optional, Type, Tuple, Dict, Any



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(PreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('PreActBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in PreActBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu(out)

        identity = out
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(out)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

class ResNetDec(nn.Module):
    def __init__(
        self,
        uplayers: List[int],
        skip_layers: List[int],
        dec_dims: List[List],
        preact: bool = False,
        pool: bool = False,
        skip_connection: bool = True,
        channel_reduction: int = 8,
        num_classes: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        final_activation: str = 'sigmoid',
        dropout_rate: float = 0,
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225],
        concat_last_feat: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.pool = pool
        self.preact = preact
        self.mean = mean
        self.std = std
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [True, True, True]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.dropout_rate = dropout_rate

        # ===== 디코더 입력 채널: concat_last_feat 기준으로 고정 =====
        in_channels =  dec_dims[-1][0]
        self.concat_last_feat = concat_last_feat
        if self.concat_last_feat==True:
            self.in_channels = in_channels//3
        else:
            self.in_channels = in_channels
        # ===== dec_dims → enc_channels/enc_spatial 맵 구성 =====
        self.skip_connection = skip_connection
        self.skip_layers = skip_layers

        if isinstance(dec_dims, dict):
            # dict 버전: 모든 skip_layers가 키로 존재해야 함
            for k in skip_layers:
                if k not in dec_dims:
                    raise ValueError(f"dec_dims dict에 skip layer {k}가 없습니다.")
            self.enc_channels: Dict[int, int] = {k: dec_dims[k][0] for k in skip_layers}
            self.enc_spatial:  Dict[int, Tuple[int,int]] = {k: (dec_dims[k][1], dec_dims[k][2]) for k in skip_layers}
        else:
            # list/tuple 버전: skip_layers와 같은 순서
            if len(dec_dims) != len(skip_layers):
                raise ValueError("dec_dims 길이는 skip_layers 길이와 같아야 합니다.")
            self.enc_channels = {k: c for k, (c, _, _) in zip(skip_layers, dec_dims)}
            self.enc_spatial  = {k: (h, w) for k, (_, h, w) in zip(skip_layers, dec_dims)}

        # ===== 입력 정리 블록 =====
        self.nin = nn.Sequential(
            conv1x1(in_channels, self.in_channels//2),
            nn.ReLU(inplace=True),
            conv1x1(self.in_channels//2, self.in_channels),
            nn.ReLU(inplace=True),
        )
        block = BasicBlock

        # ===== 디코더 스테이지 구성 =====
        self.up_layers_channel_init = self.in_channels // channel_reduction

        self.upsample_layer = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.uplayers_list = nn.ModuleList()

        # 기존 코드의 스케줄을 존중: l=0..(max(skip)-2), dec_l = max(skip)-l-1
        max_skip = max(self.skip_layers)
        for l in range(max_skip - 1):
            dec_l = max_skip - l - 1

            # 현재 디코더 피처 채널 (이전 stage의 출력)
            curr_dec_channels = self.in_channels if l == 0 else self.up_layers_channel_init // (2 ** (l - 1))

            # 스킵 연결이면 dec_dims에서 받은 정확한 채널 수를 더함
            if self.skip_connection and (dec_l in self.skip_layers):
                inplanes = curr_dec_channels + self.enc_channels[dec_l]
            else:
                inplanes = curr_dec_channels

            # 이번 스테이지 출력 채널(planes)
            planes = self.up_layers_channel_init // (2 ** l)

            self.uplayers_list.append(
                self._make_layer(block, inplanes, planes, uplayers[l], stride=1, dilate=1)
            )
        if max_skip < 2:
            last_stage_out = self.in_channels
        else:
            last_stage_out = int(self.up_layers_channel_init // (2 ** (max_skip - 2)))
        if self.skip_connection and (0 in self.skip_layers):
            conv1_in = last_stage_out + self.enc_channels[0]   # dec_dims에서 온 정확한 C
        else:
            conv1_in = last_stage_out

        conv1_out = int(self.up_layers_channel_init // (2 ** (max_skip - 1)))
        conv2_out = self.up_layers_channel_init // (2 ** (max_skip))
        
        self.conv1 = nn.Conv2d(conv1_in, conv1_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(conv1_out)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(conv2_out)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(conv2_out, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # ===== 최종 활성화 =====
        if final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif final_activation == 'leaky':
            self.final_activation = nn.LeakyReLU(inplace=True)
        else:
            self.final_activation = nn.Identity()

        self.normalize = T.Normalize(self.mean, self.std)
        self.num_classes = num_classes

        # ===== 초기화 =====
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type['BasicBlock'], inplanes: int, planes: int, blocks: int,
                    stride: int = 1, dilate: int = 1, transpose: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        resample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            if transpose:
                resample = nn.Sequential(
                    conv1x1transpose(inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                resample = nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(inplanes, planes, stride, resample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dropout_rate=self.dropout_rate))
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dropout_rate=self.dropout_rate))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _check_skip_shape(self, dec: torch.Tensor, enc: torch.Tensor, level: int):
        Hd, Wd = dec.shape[-2:]
        He, We = enc.shape[-2:]
        if (Hd, Wd) != (He, We):
            raise RuntimeError(f"Skip size mismatch at level {level}: dec {Hd,Wd} vs enc {He,We}")

    def _forward_impl(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        x: 인코더 출력 리스트(깊은 레벨이 뒤). 예:
           [enc0(B,C0,H,W), enc1, enc2, enc3(B,C3,H/8,W/8), bottleneck(B,in_channels,H/8,W/8)]
        """
        x_in = x
        x = self.nin(x_in[-1])          # bottleneck → decoder 입력
        l_skip = -2                     # enc 마지막(보틀넥 바로 앞)부터 거꾸로 사용
        max_skip = max(self.skip_layers)
        for l in range(max_skip - 1):
            dec_l = max_skip - l - 1
            x = self.upsample_layer(x)

            if self.skip_connection and (dec_l in self.skip_layers):
                self._check_skip_shape(x, x_in[l_skip], dec_l)
                x = torch.cat((x_in[l_skip], x), dim=1)
                l_skip -= 1

            x = self.uplayers_list[l](x)

        # 헤드 직전 업샘플 및 0번 스킵 처리
        x = self.upsample_layer(x)
        if self.skip_connection and (0 in self.skip_layers):
            self._check_skip_shape(x, x_in[l_skip], 0)
            x = torch.cat((x_in[l_skip], x), dim=1)

        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.upsample_layer(x)
        x = self.relu2(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.final_activation(x)
        if self.num_classes == 3:
            x = self.normalize(x)
        return x

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self._forward_impl(x)

# class ResNetDec(nn.Module):
#     def __init__(
#         self,
#         uplayers: List[int],
#         skip_layers: List[int],
#         att_layers: List[int],
#         preact: bool = False,
#         pool: bool = False,
#         skip_connection: bool = False,
#         in_channels: int = 512,
#         channel_reduction: int = 8,
#         num_classes: int = 3,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#         final_activation: str = 'sigmoid',
#         dropout_rate: float = 0,
#         mean: list = [0.485, 0.456, 0.406], 
#         std: list = [0.229, 0.224, 0.225],
#         is_pairset: bool = True, 
#         concat_last_feat: bool = True,
#     ) -> None:
#         super(ResNetDec, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.pool = pool
#         self.preact = preact
#         self.mean = mean
#         self.std = std
# #        self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [True, True, True]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.dropout_rate = dropout_rate
#         self.in_channels = in_channels

#         self.nin = nn.Sequential(
#             conv1x1(self.in_channels, self.in_channels//2),
#             nn.ReLU(inplace=True),
#             conv1x1(self.in_channels//2, self.in_channels),
#             nn.ReLU(inplace=True),
#         )
#         block = BasicBlock

#         self.skip_connection = skip_connection
#         self.skip_layers = skip_layers
#         self.concat_last_feat = concat_last_feat
#         self.up_layers_channel_init = self.in_channels//channel_reduction
#         self.upsample_layer = nn.Upsample(scale_factor = 2, mode ="bilinear", align_corners = True)
#         self.uplayers_list = nn.ModuleList()
        
#         for l in range(max(self.skip_layers)-1):
#             dec_l = max(self.skip_layers)-l-1
#             print("Decoder layer:", dec_l)
#             if l==0:
#                 if (dec_l in self.skip_layers)==True and self.skip_connection==True:
#                     if self.concat_last_feat == True:
#                         if is_pairset ==True:
#                             inplanes = self.in_channels+self.in_channels//4
#                         else:
#                             inplanes = self.in_channels+self.in_channels//2
#                     else:
#                         inplanes = self.in_channels
#                 else:
#                     inplanes = self.in_channels
#             else:
#                 if (dec_l in self.skip_layers)==True and self.skip_connection==True:
#                     if self.concat_last_feat == True:
#                         if is_pairset ==True:
#                             inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+2))
#                         else:
#                             inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+1))
#                     else:
#                         if is_pairset ==True:
#                             inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+2))
#                         else:
#                             inplanes = self.up_layers_channel_init//(2**(l-1)) + self.in_channels//(2**(l+1))
#                 else:
#                     inplanes = self.up_layers_channel_init//(2**(l-1))
#                 print(self.in_channels//(2**(l+2)))
#             planes = self.up_layers_channel_init//(2**l)
#             self.uplayers_list.append(self._make_layer(block, inplanes, planes, uplayers[l], stride=1, dilate=1))

#         if (0 in self.skip_layers)==True and self.skip_connection==True:
#             self.conv1 = nn.Conv2d(planes+self.base_width, self.up_layers_channel_init//(2**(max(self.skip_layers)-1)), kernel_size=3, stride=1, padding=1, bias=False)
#         else:
#             self.conv1 = nn.Conv2d(self.up_layers_channel_init//(2**(max(self.skip_layers)-2)), self.up_layers_channel_init//(2**(max(self.skip_layers)-1)), kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = norm_layer(self.up_layers_channel_init//(2**(max(self.skip_layers)-1)))
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(self.up_layers_channel_init//(2**(max(self.skip_layers)-1)), self.up_layers_channel_init//(2**max(self.skip_layers)), kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = norm_layer(self.up_layers_channel_init//(2**max(self.skip_layers)))
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(self.up_layers_channel_init//(2**max(self.skip_layers)), num_classes, kernel_size=3, stride=1, padding=1, bias=False)

#         if final_activation =='sigmoid':
#             self.final_activation = nn.Sigmoid()
#         elif final_activation =='softmax':
#             self.final_activation = nn.Softmax(dim=1)
#         elif final_activation =='relu':
#             self.final_activation = nn.ReLU(inplace=True)
#         elif final_activation == 'leaky':
#             self.final_activation = nn.LeakyReLU(inplace=True)
#         else:
#             self.final_activation = nn.Identity()

#         self.normalize = T.Normalize(self.mean, self.std)
#         self.num_classes = num_classes

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#        # Zero-initialize the last BN in each residual branch,
#        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

#     def _make_layer(self, block: Type[BasicBlock], inplanes: int, planes: int, blocks: int,
#                     stride: int = 1, dilate: int = 1, transpose: bool = False) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         resample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or inplanes != planes * block.expansion:
#             if transpose:
#                 resample = nn.Sequential(
#                     conv1x1transpose(inplanes, planes * block.expansion, stride),
#                     norm_layer(planes * block.expansion),
#                 )
#             else:
#                 resample = nn.Sequential(
#                   conv1x1(inplanes, planes * block.expansion, stride),
#                   norm_layer(planes * block.expansion),
#                 )

#         layers = []
#         layers.append(block(inplanes, planes, stride, resample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, dropout_rate=self.dropout_rate))
#         for _ in range(1, blocks):
#             layers.append(block(planes*block.expansion, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, dropout_rate=self.dropout_rate))

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x: List[Tensor]) -> Tensor:
#         # See note [TorchScript super()]
#         x_in = x
#         x = self.nin(x_in[-1])
#         l_skip=-2

#         for l in range(max(self.skip_layers)-1):
#             dec_l = max(self.skip_layers)-l-1
#             x = self.upsample_layer(x)
#             if (dec_l in self.skip_layers)==True and self.skip_connection==True:
#                 x = torch.cat((x_in[l_skip], x), 1)
#                 l_skip -= 1
#             else:
#                 pass
#             x = self.uplayers_list[l](x)

#         x = self.upsample_layer(x)
#         if (0 in self.skip_layers)==True and self.skip_connection==True:
#             x = torch.cat((x_in[l_skip], x), 1)
#         else:
#             pass
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)

#         x = self.upsample_layer(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)

#         x = self.conv3(x)
#         x = self.final_activation(x)
#         if self.num_classes==3:
#             x = self.normalize(x)
#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x) 


def resnet18_dec(**kwargs: Any) -> ResNetDec:
    return ResNetDec([2, 2, 1, 1], **kwargs)
