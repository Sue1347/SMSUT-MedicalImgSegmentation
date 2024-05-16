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


class UnetTrainer(BaseTrainer):
    def __init__(self, phase, args=None):
        super(UnetTrainer, self).__init__(phase, args)

    def build_network(self):
        self.net = UNet(cfg.img_channels, cfg.n_label+1, cfg.base_width, norm_type='instance', act_type='lrelu')
        self.net.to(self.device)
        # self.info(self.net)
        count_param_number(self.net, verbose=True, logger=self)
        if self.phase == 'train':
            # self.optimizer = Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.optimizer = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)

    def train_epoch(self, lb_loader, ul_loader, meter):
        self.net.train()

        lb_itr = iter(lb_loader)

        for _ in range(cfg.num_iter_per_epoch):
            bs = cfg.batch_size

            try:
                img, msk, mdl, _ = next(lb_itr)
            except StopIteration:
                lb_itr = iter(lb_loader)
                img, msk, mdl, _ = next(lb_itr)

            # print("mdl is shape like: {}".format(mdl))  # for AHDC
            img = img.to(self.device, non_blocking=True)
            msk = msk.to(self.device, non_blocking=True)

            m = mdl[0].item()

            out = self.net(img)
            # print("out is shape like : {}".format(out.shape))
            sample_loss = self.loss(out, msk)

            v, n = meter.collect_loss_by(sample_loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            self.optimizer.zero_grad()
            sample_loss.backward()
            self.optimizer.step()

            lr_ = cfg.lr * (1.0 - self.iter / (cfg.max_epoch * cfg.num_iter_per_epoch)) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
            self.iter += 1

    def saving_pseudo(self, loader_type, expr_root):
        from data_loader import inTurnLoader as inlod
        def colorize(img):
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            h, w = img.shape
            color_img = np.zeros((h, w, 3))
            for i in range(1, 5):
                bit_mask = (img == i)
                color_img[bit_mask, :] = colors[i-1][:]
            return color_img

        self.net.to(self.device)
        self.net.eval()
        # self.netF.eval()
        pred_root = pjoin(expr_root, 'pseudo')
        if not os.path.exists(pred_root):
            os.mkdir(pred_root)
        if loader_type == 'inTurn':
            loader = inlod.get_loader(cfg.png_root, 'test', 0, cfg.batch_size)
        else:
            raise NotImplementedError
        self.info(f'Predict and save in {pred_root}.')

        count = 0
        with torch.no_grad():
            for img, msk, mdl, inm in tqdm(loader):
                b, c, h, w = img.shape
                count += b
                img = img.to(self.device, non_blocking=True)

                out = self.net(img)
                pred = torch.argmax(out, dim=1)
                pred = pred.detach().cpu().numpy()
                img = img.squeeze().cpu().numpy()
                msk = msk.cpu().numpy()
                # print(" The size of pred: {}".format(pred.shape))
                # print(" The size of img: {}".format(img.shape))
                # print(" The size of img_fake: {}".format(img_fake.shape))

                for i in range(b):
                    p, m = pred[i], msk[i]
                    a = img[i]
                    p, m = colorize(p), colorize(m)
                    # print(" pred min: {} pred max: {}".format(p.astype(np.uint8).min(), p.astype(np.uint8).max()))
                    # print(" img_fake min: {} img_fake max: {}".format(b.min(), b.max()))
                    a = (a+1)*255
                    p = Image.fromarray(p.astype(np.uint8))
                    m = Image.fromarray(m.astype(np.uint8))
                    a = Image.fromarray(a).convert('RGB')
                    save_p = pjoin(pred_root, inm[i] + 'pse.jpg')
                    p.save(save_p)
                    save_m = pjoin(pred_root, inm[i] + 'gt.jpg')
                    m.save(save_m)
                    save_a = pjoin(pred_root, inm[i] + 'ori.jpg')
                    a.save(save_a)
        print(count)


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

    trainer = UnetTrainer(args.phase, args)
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