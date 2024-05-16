# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.getcwd())

import abc
import random
import numpy as np
import argparse
import time
import shutil
import logging
from os.path import join as pjoin
from medpy.metric import dc
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from trainer.baseTrainer import BaseTrainer
from network.unet import UNet
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml, get_mo_matrix
from misc.select_used_modules import make_closure_fast
from misc.visualize import count_param_number
from misc.loss import DiceAndCrossEntropyLoss
from data_loader import inTurnLoader as inlod

import warnings
warnings.filterwarnings("ignore")

class meanTeacherTrainer(BaseTrainer):
    def __init__(self, phase, args=None):
        super(meanTeacherTrainer, self).__init__(phase, args)

        self.lambda_semi = 1  # used to be 0.1
        self.ema_decay = 0.99
        self.epoch_rampup = 30  # after 5 epochs, lambda_semi get it's high values
        self.alpha = 0 # for seeing alpha trends

        self.log_step = 50

    def build_network(self):
        self.net = UNet(cfg.img_channels, cfg.n_label + 1, cfg.base_width, norm_type='instance', act_type='lrelu')
        self.net.to(self.device)
        # self.info(self.net)
        count_param_number(self.net, verbose=True, logger=self)
        if self.phase == 'train':
            self.ema = UNet(cfg.img_channels, cfg.n_label + 1, cfg.base_width, norm_type='instance', act_type='lrelu')
            for param in self.ema.parameters():
                param.detach_()
            self.ema.to(self.device)

            # self.optimizer = Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.optimizer = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)

    def update_ema_variable(self):
        if self.iter < 100:
            self.alpha = 0
        else:
            self.alpha = min(1 - 1 / (self.iter + 1), self.ema_decay) # after 10000 iters it will slowly become 0.99?
        for ema_param, param in zip(self.ema.parameters(), self.net.parameters()):
            ema_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)

    def train_epoch(self, lb_loader, ul_loader, meter):
        self.net.train()

        lb_itr = iter(lb_loader)
        ul_itr = iter(ul_loader)

        lambda_semi = self.lambda_semi * self.sigmoid_rampup(self.epoch, self.epoch_rampup)
        # the lambda for semi is too small, let's try 1, 10

        for i in range(cfg.num_iter_per_epoch):
            bs = cfg.batch_size

            try:
                img1, msk, mdl1, _ = next(lb_itr)
            except StopIteration:
                lb_itr = iter(lb_loader)
                img1, msk, mdl1, _ = next(lb_itr)

            try:
                img2, _, mdl2, _ = next(ul_itr)
            except StopIteration:
                ul_itr = iter(ul_loader)
                img2, _, mdl2, _ = next(ul_itr)

            img = torch.cat([img1, img2], dim=0)
            mdl = torch.cat([mdl1, mdl2], dim=0)
            # assert torch.all(mdl1 == mdl2), f'{mdl1}, {mdl2}'
            # assert len(torch.unique(mdl)) == 1

            img = img.to(self.device, non_blocking=True)
            ul_img = img[bs:]
            msk = msk.to(self.device, non_blocking=True)
            m = mdl[0].item()

            # noise = torch.clamp(torch.randn_like(ul_img) * 0.1, -0.2, 0.2)
            noise = torch.clamp(torch.randn_like(ul_img) * 0.01, -0.02, 0.02)
            ema_inputs = ul_img + noise
            # ema_inputs = ul_img
            '''
            1208 try same picture : saml--78.0
            1209 try a little noise: saml--
            '''

            out = self.net(img)
            out_soft = torch.softmax(out, dim=1)
            with torch.no_grad():
                ema_outputs = self.ema(ema_inputs)
                ema_outputs_soft = torch.softmax(ema_outputs, dim=1)

            sample_loss = self.loss(out[:bs], msk)
            v, n = meter.collect_loss_by(sample_loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            if self.iter < 100:
                # I think this need to be more than 1000 iters to train labeled data in the beginning
                # 150 * 200 = 30000 then maybe the first 20 epochs? 3000?
                semi_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
            else:
                semi_loss = torch.mean(
                    (out_soft[bs:] - ema_outputs_soft) ** 2)

            # semi_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
            total_loss = sample_loss + lambda_semi * semi_loss
            # 1107 不要semi_loss看看。，，，老师总是教坏的。。。
            # semi_loss: 0.0010, seg_loss: 0.0603, lambda_semi: 10.000000
            # 即，0.01 +0.06 = 0.07

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.update_ema_variable()

            if (i + 1) % self.log_step == 0:
                self.info(f'Iter %d, global_iter: %d, semi_loss: %.4f, seg_loss: %.4f, lambda_semi: %f \
                          self.alpha: %f' %
                          (i, self.iter, semi_loss.item(), sample_loss.item(), lambda_semi, self.alpha))
                self.info(f'self.alpha: %f' % self.alpha)

            lr_ = cfg.lr * (1.0 - self.iter / (cfg.max_epoch * cfg.num_iter_per_epoch)) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
            self.iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', type=str, choices=('train', 'test', 'pseudo'))
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-nm', '--expr_name', type=str)

    parser.add_argument('-i', '--model_id', type=str, help='only for test')
    parser.add_argument('-wh', '--which_ckpt', type=str, default='last')
    args = parser.parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    trainer = meanTeacherTrainer(args.phase, args)
    if args.phase == 'train':
        trainer.fit('inTurn')
    elif args.phase == 'test':
        trainer.load_model(args.model_id, args.which_ckpt)
        expr_root = pjoin(trainer.expr_root, args.model_id)
        trainer.test('inTurn', expr_root)
    elif args.phase == 'pseudo':
        trainer.load_model(args.model_id, args.which_ckpt)
        expr_root = pjoin(trainer.expr_root, args.model_id)
        trainer.saving_pseudo('inTurn', expr_root)
    else:
        raise NotImplementedError

