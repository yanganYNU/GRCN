# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:21:28 2022

@author: yang an
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d
from utils import ST_BLOCK_5  # GRCN

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
输入x-> [ batch数, 通道数, 节点数, 时间],
"""
class GRCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(GRCN, self).__init__()
        tem_size = week + day + recent
        self.block1 = ST_BLOCK_5(c_in, c_out, num_nodes, recent, K, Kt)
        self.block2 = ST_BLOCK_5(c_out, c_out, num_nodes, recent, K, Kt)
        tem_size = week + day + recent
        self.tem_size = tem_size
        self.bn = BatchNorm2d(c_in, affine=False)
        self.conv1 = Conv2d(c_out, 12, kernel_size=(1, recent),
                            stride=(1, 1), bias=True)

    def forward(self, x_w, x_d, x_r, supports):
        x_r = self.bn(x_r)
        x = x_r
        shape = x.shape

        x = self.block1(x, supports)
        x = self.block2(x, supports)
        x = self.conv1(x).squeeze().permute(0, 2, 1).contiguous()
        return x, supports, supports



