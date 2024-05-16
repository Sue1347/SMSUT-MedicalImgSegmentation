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


class crossPseTrainer(BaseTrainer):
    def __init__(self, phase, args=None):
        super(crossPseTrainer, self).__init__(phase, args)

        self.lambda_semi = 0.1
        self.log_step = 50

    def build_network(self):
        self.net = UNet(cfg.img_channels, cfg.n_label + 1, cfg.base_width, norm_type='instance', act_type='lrelu')
        self.net2 = UNet(cfg.img_channels, cfg.n_label + 1, cfg.base_width, norm_type='instance', act_type='lrelu')
        self.net.to(self.device)
        self.net2.to(self.device)
        # self.info(self.net)
        count_param_number(self.net, verbose=True, logger=self)
        count_param_number(self.net2, verbose=True, logger=self)

        if self.phase == 'train':
            # self.optimizer = Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.optimizer1 = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            self.optimizer2 = SGD(self.net2.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)

    @staticmethod
    def entropy_loss(p):
        _, C, _, _ = p.size()
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
             torch.tensor(np.log(C), device=p.device)
        ent = torch.mean(y1)
        return ent

    def crossPse_loss(pred, gt):
        _, C, _, _ = p.size()
        y = -1 * torch.sum(gt * torch.log(pred + 1e-6), dim=1) / \
             torch.tensor(np.log(C), device=p.device)
        ent = torch.mean(y)
        return ent

    def train_epoch(self, lb_loader, ul_loader, meter):
        self.net.train()
        self.net2.train()

        lb_itr = iter(lb_loader)
        ul_itr = iter(ul_loader)

        lambda_semi = self.lambda_semi * self.sigmoid_rampup(self.epoch, cfg.max_epoch)

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
            msk = msk.to(self.device, non_blocking=True)
            m = mdl[0].item()

            # supervision:
            out1 = self.net(img)
            # out1_soft = torch.softmax(out1, dim=1)
            sample1_loss = self.loss(out1[:bs], msk)
            v, n = meter.collect_loss_by(sample1_loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            out2 = self.net2(img)
            # out2_soft = torch.softmax(out2, dim=1)
            sample2_loss = self.loss(out2[:bs], msk)
            v, n = meter.collect_loss_by(sample2_loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            # semi supervision:
            pred1 = torch.argmax(out1[bs:], dim=1)
            pred1 = pred1.detach()#.cpu().numpy().astype(np.uint8)
            pred2 = torch.argmax(out2[bs:], dim=1)
            pred2 = pred2.detach()#.cpu().numpy().astype(np.uint8)
            semi1_loss = self.loss(out1[bs:], pred2)
            semi2_loss = self.loss(out2[bs:], pred1)

            total_loss = sample1_loss + sample2_loss + lambda_semi * semi1_loss + lambda_semi * semi2_loss

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            total_loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            if (i + 1) % self.log_step == 0:
                self.info(f'Iter %d, global_iter: %d, crossPse1_loss: %.4f, crossPse2_loss: %.4f, '
                          f'seg1_loss: %.4f, seg2_loss: %.4f, lambda_semi: %f' % \
                          (i, self.iter, semi1_loss.item(), semi2_loss.item(), sample1_loss.item(), sample2_loss.item(),
                           lambda_semi))

            lr_ = cfg.lr * (1.0 - self.iter / (cfg.max_epoch * cfg.num_iter_per_epoch)) ** 0.9
            for param_group in self.optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in self.optimizer2.param_groups:
                param_group['lr'] = lr_
            self.iter += 1

    def fit(self, loader_type):# 循环的训练模型
        tic = time.time()
        # Create data loaders and load npys for evaluation.
        if loader_type == 'inTurn':
            train_lb_loader = inlod.get_loader(cfg.base_root, 'train', self.fold, cfg.batch_size, cfg.data_aug)
            train_ul_loader = inlod.get_loader(cfg.base_root, 'val', self.fold, cfg.batch_size, cfg.data_aug)
            test_loader = inlod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
        elif loader_type == 'base':
            train_lb_loader = bslod.get_loader(cfg.base_root, 'train', self.fold, cfg.batch_size, cfg.data_aug)
            train_ul_loader = bslod.get_loader(cfg.base_root, 'val', self.fold, cfg.batch_size, cfg.data_aug)
            test_loader = bslod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
        else:
            raise NotImplementedError

        self.info(f'train labeled images: {train_lb_loader.dataset.__len__()}')
        self.info(f'train unlabel images: {train_ul_loader.dataset.__len__()}')
        self.info(f'test  images: {test_loader.dataset.__len__()}')

        n_tst_slic, tst_npys = get_label_npys(cfg.base_root, self.modality, 'test')
        self.info('Load data cost %.4fs.' % (time.time() - tic))
        tic = time.time()

        # Meter for recording loss and dice.
        min_better_keys = [f'loss_{i}' for i in range(cfg.n_modal)] + ['loss']
        max_better_keys = [f'dice_{i}' for i in range(cfg.n_modal)] + ['dice']
        train_meter = Meter(min_better_keys=min_better_keys, max_better_keys=[], alpha=cfg.exp_alpha)
        test_meter = Meter(min_better_keys=min_better_keys, max_better_keys=max_better_keys, alpha=1.)
        best_epoch = -1
        for epoch in range(cfg.max_epoch):
            # train stage.
            train_meter.reset_cur()
            self.train_epoch(train_lb_loader, train_ul_loader, train_meter)
            self.epoch += 1
            train_meter.update_cur()
            # train logs.
            lr1 = self.optimizer1.param_groups[0]['lr']
            lr2 = self.optimizer2.param_groups[0]['lr']
            self.info('')
            self.info(f'lr: {lr1}.')
            self.info('[TRN] Epoch: %d(%d)/%d, elapsed: %.2fs,' % (epoch, best_epoch, cfg.max_epoch, time.time() - tic)
                  + str(train_meter))
            for k, v in train_meter.cur_values.items():
                if '_' in k:  # change format like `loss_1` to `loss_t1in`
                    typ, m = k.split('_')
                    new_k = f'{typ}_{cfg.Modality(int(m)).name}'
                else:
                    new_k = k
                self.writer.add_scalar(f'train/{new_k}', v, epoch)
            self.writer.add_scalar(f'train/lr', lr1, epoch)
            tic = time.time()

            # self.scheduler.step()

            # test stage.
            test_meter.reset_cur()
            n_prd_slic, prd_npys = self.validate_epoch(test_loader, tst_npys, test_meter)
            assert n_prd_slic == n_tst_slic
            v = self.validate_dice(prd_npys, tst_npys)
            test_meter.accumulate(v, {k: 1. for k in v.keys()})
            test_meter.update_cur()
            # test logs.
            self.info('[TST] Epoch: %d/%d, elapsed: %.2fs,' % (epoch, cfg.max_epoch, time.time() - tic)
                + str(test_meter))
            for k, v in test_meter.cur_values.items():
                if '_' in k:  # change format like `loss_1` to `loss_t1in`
                    typ, m = k.split('_')
                    new_k = f'{typ}_{cfg.Modality(int(m)).name}'
                else:
                    new_k = k
                self.writer.add_scalar(f'test/{new_k}', v, epoch)
            tic = time.time()

            # save model.
            if test_meter.cur_values['dice'] >= test_meter.best_values['dice']:
                self.save_model(prefix='best')
                best_epoch = epoch

        self.save_model(prefix='last')

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

    trainer = crossPseTrainer(args.phase, args)
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