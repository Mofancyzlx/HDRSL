import torch
import torch.nn as nn
from torch.nn import init


'''
there are three module 
(1) sptial attention module
(2) channel attention module
(3) spetial and channel attention module
'''


class ChannelAttention_WH(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x) # [B, C, H, W] → [B, C, 1, 1]
        avg_result = self.avgpool(x) # [B, C, H, W] → [B, C, 1, 1]
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out) # output: [B, C, 1, 1]， 用来表示每个通道（维度 C ）的重要性
        z = output * x + x
        return z, output * x


class SpatialAttention_WH(nn.Module):

    def __init__(self,
                 in_channel,
                 inter_channel=None,
                 subsample_rate=None,
                 bn_layer=True):
        super(SpatialAttention_WH, self).__init__()
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        self.subsample_rate = subsample_rate
        self.bn_layyer = bn_layer
        max_pool_layer = nn.MaxPool2d(kernel_size=(self.subsample_rate, self.subsample_rate))
        max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.g = nn.Conv2d(in_channels=self.in_channel,
                           out_channels=self.inter_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channel,
                               out_channels=self.inter_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channel,
                             out_channels=self.inter_channel,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        # patch similarity with patch
        if subsample_rate == 1:
            self.g = nn.Sequential(self.g, max_pool_2)
            self.phi = nn.Sequential(self.phi, max_pool_2)
        else:
            self.theta = nn.Sequential(max_pool_layer, self.theta)
            self.g = nn.Sequential(max_pool_layer, self.g, max_pool_2)
            self.phi = nn.Sequential(max_pool_layer, self.phi, max_pool_2)

        if self.bn_layyer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channel,
                          out_channels=self.in_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1), nn.BatchNorm2d(self.in_channel))
            nn.init.constant_(self.W[0].weight, 0)
            nn.init.constant_(self.W[0].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channel,
                               out_channels=self.in_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        '''
        :param x: [B, C, H, W]
        :return:
        '''
        B, C, H, W = x.size()
        g_x = self.g(x).view(B, self.inter_channel, -1) # [B, C, W * H]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(B, self.inter_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(B, self.inter_channel, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = torch.nn.functional.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1)
        y = y.view(B, self.inter_channel, H // self.subsample_rate, W // self.subsample_rate)
        y = nn.Upsample(mode='bilinear', scale_factor=self.subsample_rate, align_corners=True)(y)
        w_y = self.W(y)
        z = w_y + x
        return z, w_y


