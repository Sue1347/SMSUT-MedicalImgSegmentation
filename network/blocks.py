# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_norm(channels, norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        raise NotImplementedError


def get_act(act_type, inplace=True, negative=1e-2):
    if act_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(negative_slope=negative, inplace=inplace)
    else:
        raise NotImplementedError


class UpSampleAndConcat(nn.Module):
    def __init__(self, in_ch, out_ch, transposed=True):
        super(UpSampleAndConcat, self).__init__()
        if transposed:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv1x1(in_ch, out_ch)
            )
    
    def forward(self, x, skip):
        x = self.up(x)
        return torch.cat([x, skip], dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm, act, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.bn1 = get_norm(out_ch, norm)
        self.relu = get_act(act)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = get_norm(out_ch, norm)
        self.downsample = (in_ch != out_ch) 
        if self.downsample:
            self.shortcut1 = conv1x1(in_ch, out_ch)
            self.shortcut2 = get_norm(out_ch, norm)
        
    def forward(self, x):
        identity = x 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            identity = self.shortcut1(identity)
            identity = self.shortcut2(identity)
        x += identity
        x = self.relu(x)
        return x 


class BottleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', act_type='relu', stride=1):
        super(BottleBlock, self).__init__()
        assert stride in (1, 2)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = get_norm(out_channels, norm_type)
        self.relu = get_act(act_type)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = get_norm(out_channels, norm_type)
        self.stride = stride
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                get_norm(out_channels, norm_type))

    def forward(self, x):
        identity = x
        if self.stride == 2: identity = F.avg_pool2d(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride == 2: out = F.avg_pool2d(out, 2)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.stride == 2: identity = F.avg_pool2d(x, 2)  # Orz..
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, block, width=32, norm='batch', act='lrelu', **kwargs):
        super(Encoder, self).__init__()
        self.pre_conv = nn.Conv2d(in_ch, width // 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.pre_bn = get_norm(width // 2, norm)
        self.pre_relu = get_act(act)

        self.layer1 = block(width // 2, 1 * width, norm, act, **kwargs)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = block(1 * width, 2 * width, norm, act, **kwargs)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = block(2 * width, 4 * width, norm, act, **kwargs)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.layer4 = block(4 * width, 8 * width, norm, act, **kwargs)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.layer5 = block(8 * width, 16 * width, norm, act, **kwargs)
    
    def forward(self, x):
        skips = []
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)

        x = self.layer1(x); skips.append(x)
        x = self.pool1(x)
        x = self.layer2(x); skips.append(x)
        x = self.pool2(x)
        x = self.layer3(x); skips.append(x)
        x = self.pool3(x)
        x = self.layer4(x); skips.append(x)
        x = self.pool4(x)
        x = self.layer5(x)
        return x, skips


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
        self.fc = conv1x1(width, out_ch)

    def forward(self, x, skips):
        x = self.layer4(self.up4(x, skips[3]))
        x = self.layer3(self.up3(x, skips[2]))
        x = self.layer2(self.up2(x, skips[1]))
        x = self.layer1(self.up1(x, skips[0]))
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 1, 16, 16)
    n = conv3x3(1, 2, stride=2)
    y = n(x)
    print(y.shape)