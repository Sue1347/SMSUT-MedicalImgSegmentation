# reference by: https://github.com/harshm121/M3L/tree/main
# meanteachermaskstudent.py
# linearfusemaskedconsmixbatch

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
# from torch.utils.tensorboard import SummaryWriter

import config as cfg
from trainer.baseTrainer import BaseTrainer
from network.unet import UNet
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml, get_mo_matrix, get_all_matrix
from misc.select_used_modules import make_closure_fast
from misc.visualize import count_param_number
from misc.loss import DiceAndCrossEntropyLoss
from data_loader import inTurnLoader as inlod
from network.linearfusemaskedconsmixbatch.segformer import LinearFusionMaskedConsistencyMixBatch

import warnings
warnings.filterwarnings("ignore")

class M3LTrainer(BaseTrainer):
    def __init__(self, phase, args=None):
        super(M3LTrainer, self).__init__(phase, args)

        self.lambda_semi = 1  # used to be 0.1
        self.ema_decay = 0.99
        self.epoch_rampup = 30  # after 5 epochs, lambda_semi get it's high values
        self.alpha = 0 # for seeing alpha trends

        self.log_step = 50
        self.loss = DiceAndCrossEntropyLoss(weight_ce=1.0, weight_dc=0.0, batch_dice=True)

    def build_network(self):
        self.net = LinearFusionMaskedConsistencyMixBatch(num_classes=cfg.n_label + 1)
        # LinearFusionMaskedConsistencyMixBatch(args.base_model == 'mit_b2', args, num_classes = args.num_classes)
        self.net.to(self.device)
        # self.info(self.net)
        count_param_number(self.net, verbose=True, logger=self)
        if self.phase == 'train':
            self.ema = LinearFusionMaskedConsistencyMixBatch(num_classes=cfg.n_label + 1)
            for param in self.ema.parameters():
                param.detach_()
            self.ema.to(self.device)

            # self.optimizer = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            self.optimizer = Adam(self.net.parameters(), cfg.lr, [0.9, 0.999], weight_decay=cfg.weight_decay)
            
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

            img = torch.cat([img1, img2], dim=0) # might need to assert one dimension in 1 position for rgb channel
            mdl = torch.cat([mdl1, mdl2], dim=0)
            # assert torch.all(mdl1 == mdl2), f'{mdl1}, {mdl2}'
            # assert len(torch.unique(mdl)) == 1

            
            # print(img.shape)
            img = torch.cat([img,img,img],1) # add the rgb dimension
            # print(img.shape)
            # exit()

            img = img.to(self.device, non_blocking=True)
            ul_img = img[bs:]
            msk = msk.to(self.device, non_blocking=True)
            m = mdl[0].item()

            # noise = torch.clamp(torch.randn_like(ul_img) * 0.1, -0.2, 0.2)
            # noise = torch.clamp(torch.randn_like(ul_img) * 0.01, -0.02, 0.02)
            # ema_inputs = ul_img + noise
            # ema_inputs = ul_img
            """
            Now add the procedure of M3L
            Which similar to mean teacher strategy
            """
            out = self.net(img, get_sup_loss = False, gt = None, mask = True, 
                                                        range_batches_to_mask = [cfg.batch_size, 2*cfg.batch_size]) # out is a list of two copy
            # print(" out is a list?", len(out), " out is two for sup and unsep, and one for all in original set")
            out = out[0]
            # print(out.dtype)
            out_soft = torch.softmax(out, dim=1)
            with torch.no_grad():
                ema_outputs = self.ema(img, get_sup_loss = False, gt = None, mask = False)
                ema_outputs = ema_outputs[0]
                ema_outputs_soft = torch.softmax(ema_outputs, dim=1)

            sample_loss = self.loss(out[:bs], msk)
            v, n = meter.collect_loss_by(sample_loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            
            semi_loss = self.loss(out[bs:], ema_outputs_soft[bs:])

            total_loss = sample_loss + lambda_semi * semi_loss

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

    # def fit(self, loader_type):# 循环的训练模型
    #     tic = time.time()
    #     # Create data loaders and load npys for evaluation.
    #     if loader_type == 'inTurn':
    #         train_lb_loader = inlod.get_loader(cfg.base_root, 'train', self.fold, cfg.batch_size, cfg.data_aug)
    #         train_ul_loader = inlod.get_loader(cfg.base_root, 'val', self.fold, cfg.batch_size, cfg.data_aug)
    #         test_loader = inlod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
    #     else:
    #         raise NotImplementedError

    #     self.info(f'train labeled images: {train_lb_loader.dataset.__len__()}')
    #     self.info(f'train unlabel images: {train_ul_loader.dataset.__len__()}')
    #     self.info(f'test  images: {test_loader.dataset.__len__()}')

    #     n_tst_slic, tst_npys = get_label_npys(cfg.base_root, self.modality, 'test')
    #     self.info('Load data cost %.4fs.' % (time.time() - tic))
    #     tic = time.time()

    #     # Meter for recording loss and dice.
    #     min_better_keys = [f'loss_{i}' for i in range(cfg.n_modal)] + ['loss']
    #     max_better_keys = [f'dice_{i}' for i in range(cfg.n_modal)] + ['dice']
    #     train_meter = Meter(min_better_keys=min_better_keys, max_better_keys=[], alpha=cfg.exp_alpha)
    #     test_meter = Meter(min_better_keys=min_better_keys, max_better_keys=max_better_keys, alpha=1.)
    #     best_epoch = -1
    #     for epoch in range(cfg.max_epoch):
    #         # train stage.
    #         train_meter.reset_cur()
    #         self.train_epoch(train_lb_loader, train_ul_loader, train_meter)
    #         self.epoch += 1
    #         train_meter.update_cur()
    #         # train logs.
    #         lr = self.optimizer.param_groups[0]['lr']
    #         self.info('')
    #         self.info(f'lr: {lr}.')
    #         self.info('[TRN] Epoch: %d(%d)/%d, elapsed: %.2fs,' % (epoch, best_epoch, cfg.max_epoch, time.time() - tic)
    #               + str(train_meter))
    #         for k, v in train_meter.cur_values.items():
    #             if '_' in k:  # change format like `loss_1` to `loss_t1in`
    #                 typ, m = k.split('_')
    #                 new_k = f'{typ}_{cfg.Modality(int(m)).name}'
    #             else:
    #                 new_k = k
    #             self.writer.add_scalar(f'train/{new_k}', v, epoch)
    #         self.writer.add_scalar(f'train/lr', lr, epoch)
    #         tic = time.time()

    #         # test stage.
    #         test_meter.reset_cur()
    #         n_prd_slic, prd_npys = self.validate_epoch(test_loader, tst_npys, test_meter)
    #         assert n_prd_slic == n_tst_slic
    #         v = self.validate_dice(prd_npys, tst_npys)
    #         test_meter.accumulate(v, {k: 1. for k in v.keys()})
    #         test_meter.update_cur()
    #         # test logs.
    #         self.info('[TST] Epoch: %d/%d, elapsed: %.2fs,' % (epoch, cfg.max_epoch, time.time() - tic)
    #             + str(test_meter))
    #         for k, v in test_meter.cur_values.items():
    #             if '_' in k:  # change format like `loss_1` to `loss_t1in`
    #                 typ, m = k.split('_')
    #                 new_k = f'{typ}_{cfg.Modality(int(m)).name}'
    #             else:
    #                 new_k = k
    #             self.writer.add_scalar(f'test/{new_k}', v, epoch)
    #         tic = time.time()

    #         # save model.
    #         if test_meter.cur_values['dice'] >= test_meter.best_values['dice']:
    #             self.save_model(prefix='best')
    #             best_epoch = epoch

    #     self.save_model(prefix='last')

    def validate_epoch(self, loader, npys, meter=None, save_path=None):
        self.net.eval()
        prd_npys, n_prd_slic = dict(), 0
        for k, v in npys.items():
            prd_npys[k] = np.zeros(v.shape, dtype=v.dtype)
        with torch.no_grad():
            for img, msk, mdl, inm in loader:
                b, c, h, w = img.shape
                if b != cfg.batch_size:
                    # As the last batch in inTurnLoader is not equal to batch size,
                    # prevent re-construct computer graph and save GPU memory.
                    pad = torch.zeros((cfg.batch_size - b, c, h, w), dtype=img.dtype)
                    img = torch.cat([img, pad], dim=0)

                assert len(torch.unique(mdl)) == 1

                img = torch.cat([img,img,img],1) # add the rgb dimension
                # print(img.shape)

                img = img.to(self.device, non_blocking=True)
                msk = msk.to(self.device, non_blocking=True)
                m = mdl[0].item()

                out = self.net(img)
                out = out[0] # slpit the two sets

                if b != cfg.batch_size: out = out[:b]

                sample_loss = self.loss(out, msk)
                if meter is not None:
                    v, n = meter.collect_loss_by(sample_loss.item(), m, img.size(0))
                    meter.accumulate(v, n)
                pred = torch.argmax(out, dim=1)
                pred = pred.detach().cpu().numpy()

                for i in range(b):
                    # if save_path is not None:
                    #     slic = pred[i]
                    #     slic = Image.fromarray(slic.astype(np.uint8))
                    #     save_p = pjoin(save_path, f'{inm[i]}.png')
                    #     slic.save(save_p)
                    m, pid, z = inm[i].split('_')
                    prd_npys[f'{m}_{pid}'][int(z)] = pred[i]
                    n_prd_slic += 1
        return n_prd_slic, prd_npys

    def validate_dice(self, prd_npys, gt_npys):
        mo_matrix = get_mo_matrix(prd_npys, gt_npys)
        dices = dict()
        for i in range(cfg.n_modal):
            dices[f'dice_{i}'] = mo_matrix[i, -1]
        dices['dice'] = mo_matrix[-1, -1]
        return dices

    # def test(self, loader_type, expr_root):
    #     self.net.eval()
    #     pred_root = pjoin(expr_root, 'result')
    #     # predict and save as *.png
    #     if loader_type == 'inTurn':
    #         test_loader = inlod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
    #     else:
    #         raise NotImplementedError
    #     self.info(f'Predict and save in {pred_root}.')

    #     n_gt_slic, gt_npys = get_label_npys(cfg.base_root, self.modality, 'test')
    #     prd_npys, n_prd_slic = dict(), 0
    #     for k, v in gt_npys.items():
    #         prd_npys[k] = np.zeros(v.shape, dtype=v.dtype)
    #     n_prd_slic, prd_npys = self.validate_epoch(test_loader, gt_npys, None, save_path=pred_root)

    #     # check that no prediction is missing and load ground truth npy.
    #     assert n_prd_slic == n_gt_slic
    #     # collect dice scores in a matrix, where matrix[i][j], i == modality, j == organ.
    #     # the last row and col store the mean values.
    #     matrix = get_mo_matrix(prd_npys, gt_npys)
    #     dc_matrix, hd_matrix, assd_matrix = get_all_matrix(prd_npys, gt_npys) # asd_matrix is omitted

    #     log = ''
    #     row, col = matrix.shape
    #     for i in range(row):
    #         for j in range(col):
    #             log += '%.4f' % matrix[i][j]
    #             if j != col - 1:
    #                 log += ','
    #         log += '\n'

    #     log += '\n'

    #     for i in range(row):
    #         for j in range(col):
    #             log += '%.4f' % assd_matrix[i][j]
    #             if j != col - 1:
    #                 log += ','
    #         log += '\n'
    #     save_path = pjoin(expr_root, f'{self.modality}_trois_matrix.csv')
    #     with open(save_path, 'w') as f:
    #         f.write(log)

    #     self.info(log)


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

    trainer = M3LTrainer(args.phase, args)
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

