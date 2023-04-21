# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

###GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self, c_in, c_out, K, Kt):
        super(gcn_conv_hop, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv1d(c_in_new, c_out, kernel_size=1,
                            stride=1, bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out


class ST_BLOCK_5(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_5, self).__init__()
        self.gcn_conv = gcn_conv_hop(c_out + c_in, c_out * 4, K, 1)
        self.c_out = c_out
        self.tem_size = tem_size

    def forward(self, x, supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        out = []

        for k in range(self.tem_size):
            input1 = x[:, :, :, k]
            tem1 = torch.cat((input1, h), 1)
            fea1 = self.gcn_conv(tem1, supports)
            i, j, f, o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c = c * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
            new_h = torch.tanh(new_c) * (torch.sigmoid(o))
            c = new_c
            h = new_h
            out.append(new_h)
        x = torch.stack(out, -1)
        return x
