# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from network.blocks import conv1x1, UpSampleAndConcat, BasicBlock, Encoder


class Decoder(nn.Module):
    def __init__(self, out_ch, block, width=32, norm='batch', act='lrelu', **kwargs):
        super(Decoder, self).__init__()
        self.up4 = UpSampleAndConcat(16 * width, 8 * width)
        self.layer4 = block(16 * width, 8 * width, norm, act, **kwargs)
        self.up3 = UpSampleAndConcat(8 * width, 4 * width)
        self.layer3 = block(8 * width, 4 * width, norm, act, **kwargs)
        self.up2 = UpSampleAndConcat(4 * width, 2 * width)
        self.layer2 = block(4 * width, 2 * width, norm, act, **kwargs)
        self.up1 = UpSampleAndConcat(2 * width, 1 * width)
        self.layer1 = block(2 * width, 1 * width, norm, act, **kwargs)
        self.fc1 = nn.Sequential(
            conv1x1(width, out_ch),
            nn.Tanh()
        )
        self.fc2 = conv1x1(width, out_ch)

    def forward(self, x, skips):
        x = self.layer4(self.up4(x, skips[3]))
        x = self.layer3(self.up3(x, skips[2]))
        x = self.layer2(self.up2(x, skips[1]))
        x = self.layer1(self.up1(x, skips[0]))
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2


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

    x = torch.randn(1, 1, 256, 256)
    y1, y2 = unet(x)
    print(y1.size(), y2.size())
