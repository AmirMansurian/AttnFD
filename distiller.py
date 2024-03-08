import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        self.t_net = t_net
        self.s_net = s_net
        self.args = args

    def forward(self, x):

        t_feats, t_attens, t_out = self.t_net.extract_feature(x)
        s_feats, s_attens, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)
        
        loss_attnfd = 0
        for i in range(3, feat_num):
            b,c,h,w = t_attens[i-3].shape
            
            s_attens[i-3] = self.Connectors[i](s_attens[i-3])
            loss_attnfd += (s_attens[i-3] / torch.norm(s_attens[i-3], p = 2) - t_attens[i-3] / torch.norm(t_attens[i-3], p = 2)).pow(2).sum() / (b)
        
        loss_attnfd = loss_attnfd * self.args.attn_lambda

        return s_out, loss_attnfd
