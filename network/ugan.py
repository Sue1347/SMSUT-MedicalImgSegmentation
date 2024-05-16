# -*- coding: utf-8 -*-
"""
Unified Translation and Segmentation
"""
import sys
import os

sys.path.append(os.getcwd())

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.blocks import conv3x3, conv1x1, UpSampleAndConcat, get_norm, get_act, BasicBlock, BottleBlock
import network.networks as networks  # sue changed
import config as cfg
from network.patchnce import PatchNCELoss  # sue changed


class Encoder(nn.Module):
    def __init__(self, in_ch, base_width=32, norm_type='batch', act_type='relu'):
        super(Encoder, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, base_width // 2, kernel_size=5, stride=1, padding=2, bias=False),
            get_norm(base_width // 2, norm_type),
            get_act(act_type)
        )
        self.enc1 = BasicBlock(base_width // 2, base_width, norm_type, act_type)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # x2
        self.enc2 = BasicBlock(base_width, 2 * base_width, norm_type, act_type)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # x4
        self.enc3 = BasicBlock(2 * base_width, 4 * base_width, norm_type, act_type)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # x8
        self.enc4 = BasicBlock(4 * base_width, 8 * base_width, norm_type, act_type)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # x16

    def forward(self, x):
        retn = []
        x = self.pre(x)
        e1 = self.enc1(x);
        retn.append(e1)
        ep1 = self.pool1(e1)
        e2 = self.enc2(ep1);
        retn.append(e2)
        ep2 = self.pool2(e2)
        e3 = self.enc3(ep2);
        retn.append(e3)
        ep3 = self.pool3(e3)
        e4 = self.enc4(ep3);
        retn.append(e4)
        ep4 = self.pool4(e4)
        retn.reverse()
        return ep4, retn


class Decoder(nn.Module):
    def __init__(self, out_ch, base_width=32, norm_type='batch', act_type='relu', tranposed=True, use_tanh=False):
        super(Decoder, self).__init__()
        self.up4 = UpSampleAndConcat(16 * base_width, 8 * base_width, transposed=tranposed)  # x8
        self.dec4 = BasicBlock(16 * base_width, 8 * base_width, norm_type, act_type)
        self.up3 = UpSampleAndConcat(8 * base_width, 4 * base_width, transposed=tranposed)  # x4
        self.dec3 = BasicBlock(8 * base_width, 4 * base_width, norm_type, act_type)
        self.up2 = UpSampleAndConcat(4 * base_width, 2 * base_width, transposed=tranposed)  # x2
        self.dec2 = BasicBlock(4 * base_width, 2 * base_width, norm_type, act_type)
        self.up1 = UpSampleAndConcat(2 * base_width, base_width, transposed=tranposed)  # x1
        self.dec1 = BasicBlock(2 * base_width, base_width, norm_type, act_type)

        self.fc = nn.Conv2d(base_width, out_ch, 1, bias=True)
        self.tanh = None
        if use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, e5, x_ens):
        d4 = self.dec4(self.up4(e5, x_ens[0]))
        d3 = self.dec3(self.up3(d4, x_ens[1]))
        d2 = self.dec2(self.up2(d3, x_ens[2]))
        d1 = self.dec1(self.up1(d2, x_ens[3]))
        out = self.fc(d1)
        if self.tanh is not None:
            out = self.tanh(out)
        return out


class UGAN(nn.Module):
    def __init__(self, in_ch, out_ch, n_modal, base_width=32):
        super(UGAN, self).__init__()
        self.n_modal = n_modal
        self.tsl_encoder = Encoder(in_ch + n_modal, base_width, norm_type='instance', act_type='lrelu')
        self.seg_encoder = Encoder(in_ch, base_width, norm_type='instance', act_type='lrelu')

        self.enc5 = BasicBlock(8 * base_width, 16 * base_width, norm='instance', act='lrelu')

        self.tsl_decoder = Decoder(1, base_width, norm_type='instance', act_type='lrelu',
                                   tranposed=False, use_tanh=True)
        self.seg_decoder = Decoder(out_ch, base_width, norm_type='instance', act_type='lrelu',
                                   tranposed=True, use_tanh=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m=None):
        if m is None:
            m = torch.zeros(x.size(0), self.n_modal).to(x.device)
        m = m.view(m.size(0), m.size(1), 1, 1)
        m = m.repeat(1, 1, x.size(2), x.size(3))
        seg_inputs = x
        tsl_inputs = torch.cat([x, m], dim=1)

        tsl_out, tsl_ens = self.tsl_encoder(tsl_inputs)
        tsl_out = self.enc5(tsl_out)
        tsl = self.tsl_decoder(tsl_out, tsl_ens)

        seg_out, seg_ens = self.seg_encoder(seg_inputs)
        seg_out = self.enc5(seg_out)
        seg = self.seg_decoder(seg_out, seg_ens)
        return seg, tsl


class UGANnce(nn.Module):
    def __init__(self, in_ch, out_ch, n_modal, base_width=32, val_phase=False):
        super(UGANnce, self).__init__()
        self.n_modal = n_modal
        self.val_phase = val_phase
        self.tsl_encoder = Encoder(in_ch + n_modal, base_width, norm_type='instance', act_type='lrelu')
        self.seg_encoder = Encoder(in_ch, base_width, norm_type='instance', act_type='lrelu')

        self.enc5 = BasicBlock(8 * base_width, 16 * base_width, norm='instance', act='lrelu')

        self.netF = define_F(in_ch)  # 全用CUT默认输入
        if not self.netF.mlp_init:  # 之前忘了，一直每次调用，，，都初始化一遍。。。我傻
            self.netF.create_mlp(cfg.nce_layers)

        self.tsl_decoder = Decoder(1, base_width, norm_type='instance', act_type='lrelu',
                                   tranposed=False, use_tanh=True)
        self.seg_decoder = Decoder(out_ch, base_width, norm_type='instance', act_type='lrelu',
                                   tranposed=True, use_tanh=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m=None, sample_ids=None, val_phase=False):
        if m is None:
            m = torch.zeros(x.size(0), self.n_modal).to(x.device)
        m = m.view(m.size(0), m.size(1), 1, 1)
        m = m.repeat(1, 1, x.size(2), x.size(3))
        seg_inputs = x
        tsl_inputs = torch.cat([x, m], dim=1)

        tsl_out, tsl_ens = self.tsl_encoder(tsl_inputs)
        # tsl_out is a vector:16 128 16 16
        tsl_out_1 = self.enc5(tsl_out)
        # tsl_out is a vector:16 256 16 16
        tsl = self.tsl_decoder(tsl_out_1, tsl_ens)

        seg_out, seg_ens = self.seg_encoder(seg_inputs)
        seg_out = self.enc5(seg_out)
        seg = self.seg_decoder(seg_out, seg_ens)

        if val_phase:
            return seg, tsl

        # tsl_out_f = tsl_out_1
        # tsl_out_f = tsl_out.reshape(tsl_out.shape[0], tsl_out.shape[1] * tsl_out.shape[2] * tsl_out.shape[3])
        # when there is two 0, it involves two part features.

        feats = []

        # feats.append(tsl_out)
        feats.append(tsl_out_1)
        # for layer in cfg.nce_layers: #用成了channels, eg:[0,85,170,255]
        #     feat = tsl_out_1[:, layer, :, :]
        #     # print("******************************")
        #     # print(" feat has shape of {} {} {}".format(feat.shape[0], feat.shape[1], feat.shape[2]))# 16,16,16
        #     feat = feat.unsqueeze(1)
        #     # 16,1,16,16
        #     feats.append(feat)

        if sample_ids is None:
            feat_pool, sample_ids = self.netF(feats)
        else:
            feat_pool, _ = self.netF(feats, patch_ids=sample_ids)

        return seg, tsl, feat_pool, sample_ids


class Discriminator(nn.Module):
    def __init__(self, input_size, n_modal, base_width=32, max_width=512):
        super(Discriminator, self).__init__()
        blocks = []
        blocks += [nn.Conv2d(1, base_width, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(inplace=True)]

        repeat_num = int(np.log2(input_size)) - 2
        in_width = base_width
        for _ in range(1, repeat_num):
            out_width = min(in_width * 2, max_width)
            blocks += [BottleBlock(in_width, out_width, norm_type='instance', act_type='lrelu', stride=2)]
            in_width = out_width
        self.main = nn.Sequential(*blocks)

        kernel_size = int(input_size / np.power(2, repeat_num))
        self.conv_src = nn.Conv2d(out_width, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(out_width, n_modal, kernel_size=kernel_size, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.main(x)
        out_src = self.conv_src(out)
        out_cls = self.conv_cls(out)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


# maybe it should be here for construct the frameworks
def define_F(input_nc, netF='mlp_sample', norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02,
             no_antialias=False, gpu_ids=None, netF_nc=256):
    """
    input_nc: int
    nce_layer: int
    netF='mlp_sample', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
    no_antialias=False, gpu_ids=None
    """
    # nce_layer is len(cfg.nce_layers)
    if gpu_ids is None:
        gpu_ids = []
    net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=netF_nc)
    # net.create_mlp(feats)
    return init_net(net, init_type, init_gain, gpu_ids)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """
    Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])

    if initialize_weights:
        networks.init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = networks.Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, nce_layers, input_nc=256):  # 要提前建立网络，所以直接传递feats的channel数
        """
                for mlp_id, feat in enumerate(feats):
                    input_nc = feat.shape[1]
                    mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
                    if len(self.gpu_ids) > 0:
                        mlp.cuda()
                    setattr(self, 'mlp_%d' % mlp_id, mlp)
                init_net(self, self.init_type, self.init_gain, self.gpu_ids)
                self.mlp_init = True
                5th has 256 channels.
        """
        for mlp_id, layer in enumerate(nce_layers):
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(cfg.nce_layers)  # 可不可以提前create，否则网络里没有这个的对应字典
        # feats: list of 5 tensors.

        for feat_id, feat in enumerate(feats):
            # print("*************************************************")
            # print("feat is : {}".format(feat))
            # print("feat as tuple length: {}".format(len(feat))) # 16???

            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            # print("feat has B C H W : %d %d %d %d " %(B, C, H, W)) # 16,1,16,16
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                mlp.cuda()
                x_sample = x_sample.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                # RuntimeError: Expected all tensors to be on the same device
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


if __name__ == '__main__':
    import config as cfg
    from misc.visualize import count_param_number

    net = UGANnce(1, 5, 4, 16)
    # net = Discriminator(256, 4, 16)
    print(net)
    count_param_number(net, verbose=True)
    exit()

    count_param_number(net, verbose=True)
    x = torch.randn(1, 1, 256, 256)
    y1, y2 = net(x)
    print(y1.shape, y2.shape)
