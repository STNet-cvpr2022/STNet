# --------------------------------------------------------
# Inter-attention
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from videoanalyst.model.attention.attention_base import (TRACK_ATTENTION, VOS_ATTENTION)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.attention.attention_impl.cbam import  ChannelGate, SpatialGate, BasicConv

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

'''
Feature Separation Part
'''
class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out

class Spa_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Spa_Module, self).__init__()
        self.chanel_in = in_dim
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        proj_value = x.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Tem_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(Tem_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


@VOS_ATTENTION.register
@TRACK_ATTENTION.register

class Interatten(ModuleBase):
    """ inter attention
    """
    def __init__(self):
        super().__init__()
        in_planes = 256
        out_planes = 256
        # self.tem_attention = ChannelGate(in_planes, reduction_ratio=16, pool_types=['avg'])
        self.inter_attention = FSP(in_planes, out_planes)
        # self.spa_attention = SpatialGate()
        self.tem_attention = Tem_Module(in_planes)
        self.spa_attention = Spa_Module(in_planes)
        self.conv_cat = BasicConv(4*in_planes, out_planes, kernel_size=1, stride=1, padding=0, relu=True, bn=True, bias=False)

    def forward(self, tem_fea, spa_fea):
        tem_fea =  self.tem_attention(tem_fea)
        # spa_fea_c = self.spa_attention(spa_fea)
        # out = torch.cat([tem_fea_c, spa_fea_c, tem_fea, spa_fea], dim=1)
        # out = self.conv_cat(out)
        # out = out + tem_fea_c + spa_fea_c
        spa_fea = self.inter_attention(tem_fea, spa_fea)
        spa_fea = self.spa_attention(spa_fea)
        return spa_fea

     
