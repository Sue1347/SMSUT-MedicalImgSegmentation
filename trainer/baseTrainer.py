# -*- coding: utf-8 -*-
'''
sue的注释版。
'''
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
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml, get_mo_matrix, get_all_matrix
from misc.select_used_modules import make_closure_fast
from misc.visualize import count_param_number
from misc.loss import DiceAndCrossEntropyLoss
from data_loader import inTurnLoader as inlod
from data_loader import baseLoader as bslod


class BaseTrainer(object):
    #创建一个类，所有的trainer都基于这个类开展
    def __init__(self, phase, args=None):#初始化相应的属性
        self.args = args
        self.device = torch.device('cuda')
        self.phase = phase # test 或者 train
        self.fold = 0 if args is None else self.args.fold
        torch.backends.cudnn.benchmark = True

        maybe_mkdir(cfg.expr_root)#输出的目录
        if args is None or args.expr_name is None or len(args.expr_name) == 0:
            expr_name = self.__class__.__name__
        else:
            expr_name = args.expr_name
        expr_root = pjoin(cfg.expr_root, expr_name)
        self.expr_root, self.model_idx = expr_root, None

        self.writer = None
        self.modality = 'all'
        self.logger = None
        if self.phase == 'train':
            self.init_train_env(self.expr_root)
        self.net = None
        self.build_network()
        self.loss = DiceAndCrossEntropyLoss(weight_ce=cfg.weight_ce, weight_dc=cfg.weight_dc, batch_dice=True)

        self.info(self.args)

        self.epoch = 0
        self.iter = 0

    @staticmethod
    def sigmoid_rampup(current, rampup_length):#对loss cons的处理，指数增长，之前较少影响，之后加大比重
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def register_experiment_args(self, path, filename='expriments.log'):#记录实验数据（好酷啊啊啊啊）
        assert self.phase == 'train'
        root = pjoin(self.expr_root, self.model_idx)
        with open(pjoin(path, filename), 'a') as f:
            f.write(self.__class__.__name__ + ', ' + root + '\n')
            f.write(str(self.args) + '\n\n')

    def init_train_env(self, expr_root):# 初始化训练的环境
        maybe_mkdir(expr_root)
        self.model_idx = str(len(os.listdir(expr_root))).rjust(3, '0')
        model_root = pjoin(expr_root, self.model_idx)
        code_root = pjoin(model_root, 'code')
        ckpt_root = pjoin(model_root, 'ckpt')
        tb_root = pjoin(model_root, 'tb')
        result_root = pjoin(model_root, 'result')
        sample_root = pjoin(model_root, 'sample')
        maybe_mkdir(model_root, ckpt_root, tb_root, result_root, sample_root)# 模型，checkpoint，tb，结果，样本的目录依次建一遍
        shutil.copytree(os.getcwd(), code_root) #递归copy多个目录到指定目录下，保存本次模型的代码。这个超有用！
        self.writer = SummaryWriter(tb_root) #指示tensorboard的存放地址

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger('FileLogger')
        file_handler = logging.FileHandler(pjoin(model_root, 'train.log'), mode='a', encoding='utf-8')
        self.logger.addHandler(file_handler)

        self.info(f'Create train environment in {model_root}.')

    def info(self, s):
        if self.logger is not None:
            self.logger.info(s); return
        print(s)

    @abc.abstractmethod
    def build_network(self):# abc抽象基类，定义抽象基类的组件，来使用后续的类似继承的关系
        """
        Build networks. If phase == train, build optimizer and scheduler.
        """
        pass

    def load_model(self, model_idx=None, which_ckpt='last'):
        if model_idx is None:
            model_idx = self.model_idx
        path = pjoin(self.expr_root, model_idx, 'ckpt', f'{which_ckpt}.ckpt')
        self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.info(f'Load model from {path}.')

    def save_model(self, prefix):
        path = pjoin(self.expr_root, self.model_idx, 'ckpt', f'{prefix}.ckpt')
        torch.save(self.net.state_dict(), path)
        self.info(f'Save model to {path}.')

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
            lr = self.optimizer.param_groups[0]['lr']
            self.info('')
            self.info(f'lr: {lr}.')
            self.info('[TRN] Epoch: %d(%d)/%d, elapsed: %.2fs,' % (epoch, best_epoch, cfg.max_epoch, time.time() - tic)
                  + str(train_meter))
            for k, v in train_meter.cur_values.items():
                if '_' in k:  # change format like `loss_1` to `loss_t1in`
                    typ, m = k.split('_')
                    new_k = f'{typ}_{cfg.Modality(int(m)).name}'
                else:
                    new_k = k
                self.writer.add_scalar(f'train/{new_k}', v, epoch)
            self.writer.add_scalar(f'train/lr', lr, epoch)
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

    @abc.abstractmethod
    def train_epoch(self, lb_loader, ul_loader, meter):
        pass

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
                img = img.to(self.device, non_blocking=True)
                msk = msk.to(self.device, non_blocking=True)
                m = mdl[0].item()
                out = self.net(img)
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

    def test(self, loader_type, expr_root):
        self.net.eval()
        pred_root = pjoin(expr_root, 'result')
        # predict and save as *.png
        if loader_type == 'inTurn':
            test_loader = inlod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
        else:
            raise NotImplementedError
        self.info(f'Predict and save in {pred_root}.')

        n_gt_slic, gt_npys = get_label_npys(cfg.base_root, self.modality, 'test')
        prd_npys, n_prd_slic = dict(), 0
        for k, v in gt_npys.items():
            prd_npys[k] = np.zeros(v.shape, dtype=v.dtype)
        n_prd_slic, prd_npys = self.validate_epoch(test_loader, gt_npys, None, save_path=pred_root)

        

        # check that no prediction is missing and load ground truth npy.
        assert n_prd_slic == n_gt_slic
        # collect dice scores in a matrix, where matrix[i][j], i == modality, j == organ.
        # the last row and col store the mean values.
        matrix = get_mo_matrix(prd_npys, gt_npys)
        dc_matrix, hd_matrix, assd_matrix = get_all_matrix(prd_npys, gt_npys) # asd_matrix is omitted

        log = ''
        row, col = matrix.shape
        for i in range(row):
            for j in range(col):
                log += '%.4f' % matrix[i][j]
                if j != col - 1:
                    log += ','
            log += '\n'
        # save_path = pjoin(expr_root, f'{self.modality}_dice_matrix.csv')
        # with open(save_path, 'w') as f:
        #     f.write(log)

        log += '\n'

        # for i in range(row):
        #     for j in range(col):
        #         log += '%.4f' % hd_matrix[i][j]
        #         if j != col - 1:
        #             log += ','
        #     log += '\n'
        # # save_path = pjoin(expr_root, f'{self.modality}_hd_matrix.csv')
        # # with open(save_path, 'w') as f:
        # #     f.write(log)
        #
        # log += '\n'

        for i in range(row):
            for j in range(col):
                log += '%.4f' % assd_matrix[i][j]
                if j != col - 1:
                    log += ','
            log += '\n'
        save_path = pjoin(expr_root, f'{self.modality}_trois_matrix.csv')

        

        with open(save_path, 'w') as f:
            f.write(log)

        self.info(log)

    def saving_pseudo(self, loader_type, expr_root):
        from data_loader import inTurnLoader as inlod
        def colorize(img):
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            h, w = img.shape
            color_img = np.zeros((h, w, 3))
            for i in range(1, 5):
                bit_mask = (img == i)
                color_img[bit_mask, :] = colors[i - 1][:]
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
                    a = (a + 1) * 255
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
    pass
