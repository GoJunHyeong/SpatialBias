import torch.nn as nn
import math
import torch
import numpy as np

class Spatial_bias(nn.Module):
    def __init__(self, channel):
        super(Spatial_bias, self).__init__()

        sb_in_plane = 5
        self.reduce_r = 10
        self.num_sb = sb_in_plane - 2
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(channel, sb_in_plane, kernel_size=1, bias = False, stride = 1),
            nn.BatchNorm2d(sb_in_plane),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.reduce_r)
        )
        self.sb_conv = nn.Conv1d((self.reduce_r**2), (self.reduce_r**2), 3, padding = 0)

    def forward(self, x_):
        x_ = self.feature_reduction(x_)
        x = torch.reshape(x_, (x_.shape[0], x_.shape[1], 1, -1)).squeeze().permute(0, 2, 1)
        x = self.sb_conv(x).unsqueeze(2)
        x = x.permute(0, 3, 2, 1).reshape(x_.shape[0], self.num_sb, self.reduce_r, self.reduce_r)
        return x