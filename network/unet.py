# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from network.blocks import BasicBlock, Encoder, Decoder


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_width=64, 
                 norm_type='batch', act_type='relu'):
        super(UNet, self).__init__()
        
        self.encoder = Encoder(in_ch, BasicBlock, base_width, norm=norm_type, act=act_type)
        self.decoder = Decoder(out_ch, BasicBlock, base_width, norm=norm_type, act=act_type)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu' if act_type == 'relu' else 'leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x


if __name__ == '__main__':
    unet = UNet(1, 5, 8, norm_type='instance', act_type='relu')
    # print(unet)

    x = torch.randn(1, 1, 192, 192)
    y = unet(x)
    print(y.size())
