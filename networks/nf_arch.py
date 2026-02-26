from torch import nn
import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Callable, Union, Tuple, Iterable, List
Shape = Tuple[int]
ShapeList = List[Tuple[int]]
from .flow2d import *


# 2d-Flow network
def freia_2dflow_head(dim_in, n_coupling_blocks, kernel_size, ch_rate, leaky_slope, clamp, linear_func):
    nodes = list()
    nodes.append(InputNode(dim_in[0], dim_in[1], dim_in[2], name=f'input'))

    for k in range(n_coupling_blocks):
        node_to_permute=[]
        coupling_inputs = []
        if k == 0:
            node_to_permute.append(nodes[0].out0)
        else:
            node_to_permute.append(nodes[-1].out0)

        nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        coupling_inputs.append(nodes[-1].out0)
        if linear_func ==True:
            nodes.append(Node(coupling_inputs, parallel_glow_coupling_layer,
                              {'clamp': clamp, 'F_class': Convolutions,
                               'F_args': {'exp_ch': ch_rate, 'kernel_size': kernel_size, 'leaky_slope': leaky_slope, 'block_no': k}}, name=F'conditioner_{k}'))
        else:
            nodes.append(Node(coupling_inputs, parallel_flowpp_coupling_layer,
                              {'clamp': clamp, 'F_class': Convolutions,
                               'F_args': {'exp_ch': ch_rate, 'kernel_size': kernel_size, 'leaky_slope': leaky_slope, 'block_no': k}}, name=F'conditioner_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output'))

    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf

