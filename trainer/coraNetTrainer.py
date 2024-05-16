# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.getcwd())

import abc
import random
import numpy as np
import argparse
import time
import os
import shutil
import logging
from os.path import join as pjoin
from medpy.metric import dc
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import config as cfg
from trainer.baseTrainer import BaseTrainer
from network.unet import UNet
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml, get_mo_matrix
from misc.select_used_modules import make_closure_fast
from misc.visualize import count_param_number
from misc.loss import DiceAndCrossEntropyLoss, SoftDiceLoss
from data_loader import inTurnLoader as inlod

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

warnings.filterwarnings("ignore")


class DiceAndCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce=1., weight_dc=1., batch_dice=False, weight=cfg.default_w, reduc=False):
        super(DiceAndCrossEntropyLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dc = weight_dc
        self.dc = SoftDiceLoss(batch_dice=batch_dice)
        self.ce = nn.CrossEntropyLoss(weight=weight.cuda())
        if reduc:
            self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight.cuda())

    def forward(self, x, y):
        dc_loss = self.dc(x, y) if self.weight_dc != 0 else 0.
        ce_loss = self.ce(x, y) if self.weight_ce != 0 else 0.
        loss = self.weight_dc * dc_loss + self.weight_ce * ce_loss
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


class make_data(Dataset):
    def __init__(self, imgs, plabs, masks, labs, mdls):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]  #
        self.plab = [np.squeeze(lab) for lab in plabs]
        self.mask = [np.squeeze(mask) for mask in masks]  # np.squeeze()
        self.lab = [lab.cpu().squeeze().numpy() for lab in labs]
        self.num = len(self.img)
        self.mdl = [mdl.cpu().squeeze().numpy() for mdl in mdls]

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx], self.mdl[idx]
        imgs, plabs, masks, labs, mdl = samples
        return imgs, plabs.astype(np.long), masks.astype(np.float), labs.astype(np.long), mdl

    def __len__(self):
        return self.num


class coraNetTrainer(BaseTrainer):
    def __init__(self, phase, args=None):
        super(coraNetTrainer, self).__init__(phase, args)

        self.lambda_semi = 1  # according to UAMT codes
        self.ema_decay = 0.99
        self.epoch_rampup = 30  # according to UAMT codes
        self.alpha = 0  # for seeing alpha trends
        self.model_id = args.model_id

        self.log_step = 50
        self.loss = DiceAndCrossEntropyLoss(weight_ce=cfg.weight_ce, weight_dc=cfg.weight_dc, batch_dice=True)
        self.conloss = DiceAndCrossEntropyLoss(weight_ce=1., weight_dc=0., weight=cfg.w_con)
        self.radloss = DiceAndCrossEntropyLoss(weight_ce=1., weight_dc=0., weight=cfg.w_rad)
        self.CAceloss = DiceAndCrossEntropyLoss(weight_ce=1., weight_dc=0., reduc=True)
        self.diceloss = DiceAndCrossEntropyLoss(weight_ce=0., weight_dc=1.)
        self.CAconloss = DiceAndCrossEntropyLoss(weight_ce=1., weight_dc=0., weight=cfg.w_con, reduc=True)
        self.CAradloss = DiceAndCrossEntropyLoss(weight_ce=1., weight_dc=0., weight=cfg.w_rad, reduc=True)

    def load_ema_model(self, model_idx=None, which_ckpt='last'):
        if model_idx is None:
            model_idx = self.model_idx
        path = pjoin(self.expr_root, model_idx, 'ckpt', f'{which_ckpt}.ckpt')
        self.ema.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.info(f'Load model from {path}.')

    def save_ema_model(self, prefix):
        path = pjoin(self.expr_root, self.model_idx, 'ckpt', f'{prefix}.ckpt')
        torch.save(self.ema.state_dict(), path)
        self.info(f'Save model to {path}.')

    def get_mask(out):
        probs = F.softmax(out, 1)
        masks = (probs >= cfg.thres).float()
        masks = masks[:, 1, :, :].contiguous()
        return masks

    def softmax_mse_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = torch.softmax(input_logits, dim=1)
        target_softmax = torch.softmax(target_logits, dim=1)

        mse_loss = (input_softmax - target_softmax) ** 2
        return mse_loss

    def build_network(self):  # net has output of 3 layers.
        self.net = UNet(cfg.img_channels, cfg.n_label * 3 + 1, cfg.base_width, norm_type='instance', act_type='lrelu')
        self.net.to(self.device)
        # self.info(self.net)
        count_param_number(self.net, verbose=True, logger=self)
        if self.phase == 'train':
            self.ema = UNet(cfg.img_channels, cfg.n_label * 3 + 1, cfg.base_width, norm_type='instance',
                            act_type='lrelu')
            for param in self.ema.parameters():
                param.detach_()
            self.ema.to(self.device)

            # self.optimizer = Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.optimizer = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            # cfg.weight_decay
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)

    def update_ema_variable(self):
        if self.iter < 100:
            self.alpha = 0
        else:
            self.alpha = min(1 - 1 / (self.iter + 1), self.ema_decay)  # after 10000 iters it will slowly become 0.99?
        for ema_param, param in zip(self.ema.parameters(), self.net.parameters()):
            ema_param.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)

    @torch.no_grad()
    def pred_unlabel(self, ul_loader):
        # ul_loader is batch size == 1
        logging.info('Starting predict unlab')
        unimg, unlab, unmask, labs, mdls = [], [], [], [], []
        plab_dice = 0
        for (step, data) in enumerate(ul_loader):
            bs = cfg.batch_size
            # ul_itr = iter(ul_loader)
            img, lab, mdl2, _ = data
            img, lab = img.cuda(), lab.cuda()
            # print("mdl2 is {}".format(mdl2))  # is 0-5

            out = self.net(img)
            out_back = out[0, 0, :, :].unsqueeze(dim=0).unsqueeze(dim=1)
            # print("out back is shape like :{}".format(out_back.shape))  # [8,1,256,256]
            out_0 = out[0, 1: 1 * cfg.n_label + 1, :, :].unsqueeze(dim=0)
            # print("out 0 is shape like :{}".format(out_0.shape))
            out0 = torch.cat([out_back, out_0], dim=1)
            out_1 = out[0, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :].unsqueeze(dim=0)
            out1 = torch.cat([out_back, out_1], dim=1)
            out_2 = out[0, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :].unsqueeze(dim=0)
            out2 = torch.cat([out_back, out_2], dim=1)

            # plab0 = self.get_mask(out[0])
            plab0 = torch.argmax(out0, dim=1).cpu().numpy()
            # plab1 = self.get_mask(out[1])
            plab1 = torch.argmax(out1, dim=1).cpu().numpy()
            # plab2 = self.get_mask(out[2])
            plab2 = torch.argmax(out2, dim=1).cpu().numpy()

            mask = (plab1 == plab2).astype(np.long)
            plab = plab0
            unimg.append(img)
            unlab.append(plab)
            unmask.append(mask)
            labs.append(lab)
            mdls.append(mdl2)

            lab = lab.cpu().numpy()
            plab_dice += dc(plab, lab)
        plab_dice /= len(ul_loader)
        logging.info('Pseudo label dice : {}'.format(plab_dice))
        """
        加一个模态？为什么输出的channel是8？ 因为ul loader的batch是8
        为什么dice值变小了？大概是1/2？是len还是count？
        """
        # print("unimg is shape: {},  size: {}".format(len(unimg), unimg[0].size))
        new_loader = DataLoader(make_data(unimg, unlab, unmask, labs, mdls), batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, drop_last=True)
        return new_loader, plab_dice

    def train_epoch(self, lb_loader, ul_loader, new_loader, meter):
        # consistency_criterion = self.softmax_mse_loss()

        train_loss, train_loss1, train_loss2, train_loss3, train_dice, unlab_dice = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        loss_ce1_, loss_con1_, loss_rad1_, dice_loss1_, loss_ce2_, loss_con2_, loss_rad2_, dice_loss2_ = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        lab_rad_dice, unlab_rad_dice, lab_con_dice, unlab_con_dice = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        lb_itr = iter(lb_loader)
        ul_itr = iter(ul_loader)
        pse_itr = iter(new_loader)

        self.net.train()
        self.ema.train()

        for i in range(cfg.num_iter_per_epoch):
            bs = cfg.batch_size

            # get source data
            try:
                img1, msk, mdl1, _ = next(lb_itr)
            except StopIteration:
                lb_itr = iter(lb_loader)
                img1, msk, mdl1, _ = next(lb_itr)

            try:
                _, _, mdl2, _ = next(ul_itr)
            except StopIteration:
                ul_itr = iter(ul_loader)
                _, _, mdl2, _ = next(ul_itr)

            try:
                img2, plab2, mask, _, mdl2 = next(pse_itr)
            except StopIteration:
                pse_itr = iter(new_loader)
                img2, plab2, mask, _, mdl2 = next(pse_itr)

            img1, msk, mdl1 = img1.cuda(), msk.cuda(), mdl1.cuda()
            img2, plab2, mask, mdl2 = img2.cuda(), plab2.cuda(), mask.cuda(), mdl2.cuda()
            img2 = img2.unsqueeze(dim=1)
            # plab2 = plab2.unsqueeze(dim=1)
            # mask = mask.unsqueeze(dim=1)
            # print("img1 has shape of {}".format(img1.shape))  ### [8, 1, 256, 256]
            # print("label has shape of {}".format(msk.shape))
            # print("img2 has shape of {}".format(img2.shape))  ### [8, 1, 256, 256]
            # print("plab2 has shape of {}".format(plab2.shape))  ### [8, 256, 256]
            # print("mask has shape of {}".format(mask.shape))  ### [8, 256, 256]

            img = torch.cat([img1, img2], dim=0)
            mdl = torch.cat([mdl1, mdl2], dim=0)
            img = img.to(self.device, non_blocking=True)
            ul_img = img[bs:]
            msk = msk.to(self.device, non_blocking=True)
            m = mdl[0].item()
            noise = torch.clamp(torch.randn_like(ul_img) * 0.01, -0.02, 0.02)
            ema_inputs = ul_img + noise

            '''Supervised Loss'''
            out_s = self.net(img1)
            out_back = out_s[:bs, 0, :, :].unsqueeze(dim=1)
            out_0 = out_s[:bs, 1: 1 * cfg.n_label + 1, :, :]
            out0 = torch.cat([out_back, out_0], dim=1)
            out_1 = out_s[:bs, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :]
            out1 = torch.cat([out_back, out_1], dim=1)
            out_2 = out_s[:bs, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :]
            out2 = torch.cat([out_back, out_2], dim=1)

            # loss
            cedc_loss = self.loss(out0, msk)
            loss_con = self.conloss(out1, msk)
            loss_rad = self.radloss(out2, msk)
            supervised_loss = (cedc_loss + loss_con + loss_rad) / 4

            '''Certain Areas'''
            out_p = self.net(img2)
            out_back = out_p[:bs, 0, :, :].unsqueeze(dim=1)
            out_20 = out_p[:bs, 1: 1 * cfg.n_label + 1, :, :]
            out20 = torch.cat([out_back, out_20], dim=1)
            out_21 = out_p[:bs, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :]
            out21 = torch.cat([out_back, out_21], dim=1)
            out_22 = out_p[:bs, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :]
            out22 = torch.cat([out_back, out_22], dim=1)

            # dice_loss2 = DICE(out2[0], plab2, mask)
            dice_loss2 = self.diceloss(out20, plab2)
            loss_ce2 = (self.CAceloss(out20, plab2) * mask).sum() / (mask.sum() + 1e-16)  #
            loss_con2 = (self.CAradloss(out21, plab2) * mask).sum() / (mask.sum() + 1e-16)  #
            loss_rad2 = (self.CAconloss(out22, plab2) * mask).sum() / (mask.sum() + 1e-16)  #

            certain_loss = (loss_ce2 + dice_loss2) / 2

            '''Uncertain Areas---Mean Teacher'''
            mask = (1 - mask).unsqueeze(1)
            with torch.no_grad():
                out_ema = self.ema(img2)
            ema_back = out_ema[:bs, 0, :, :].unsqueeze(dim=1)
            ema_0 = out_ema[:bs, 1: 1 * cfg.n_label + 1, :, :]
            ema0 = torch.cat([ema_back, ema_0], dim=1)
            ema_1 = out_ema[:bs, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :]
            ema1 = torch.cat([ema_back, ema_1], dim=1)
            ema_2 = out_ema[:bs, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :]
            ema2 = torch.cat([ema_back, ema_2], dim=1)

            # consistency_weight = self.consistency * self.get_current_consistency_weight(self.epoch)
            consistency_weight = self.lambda_semi * self.sigmoid_rampup(self.epoch, self.epoch_rampup)
            consistency_dist1 = self.softmax_mse_loss(out20, ema0)
            const_loss1 = consistency_weight * ((consistency_dist1 * mask).sum() / (mask.sum() + 1e-16))
            consistency_dist2 = self.softmax_mse_loss(out21, ema1)
            const_loss2 = consistency_weight * ((consistency_dist2 * mask).sum() / (mask.sum() + 1e-16))
            consistency_dist3 = self.softmax_mse_loss(out22, ema2)
            const_loss3 = consistency_weight * ((consistency_dist3 * mask).sum() / (mask.sum() + 1e-16))
            uncertain_loss = (const_loss1 + const_loss2 + const_loss3) / 3
            # logging.info(uncertain_loss)

            if self.iter < 1000: # 20220515 for better results...
                certain_loss = torch.tensor(0., device=self.device)
                uncertain_loss = torch.tensor(0., device=self.device)
            loss = supervised_loss + certain_loss + uncertain_loss * 0.1  # uncertain_loss * 0.3 #+ certain_loss*0.5

            # supervised loss
            v, n = meter.collect_loss_by(loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_ema_variable()

            ### change tonight!!!!
            # with torch.no_grad():
            #     mask1 = get_mask(out1[0])
            #     mask2 = get_mask(out2[0])
            #
            #     train_dice.update(statistic.dice_ratio(mask1, lab1), 1)
            #     unlab_dice.update(statistic.dice_ratio(mask2, lab2), 1)
            #
            #     lab_rad_dice.update(statistic.dice_ratio(get_mask(out1[2]), lab1), 1)
            #     unlab_rad_dice.update(statistic.dice_ratio(get_mask(out2[2]), lab2), 1)
            #     lab_con_dice.update(statistic.dice_ratio(get_mask(out1[1]), lab1), 1)
            #     unlab_con_dice.update(statistic.dice_ratio(get_mask(out2[1]), lab2), 1)
            #
            #     train_loss.update(loss.item(), 1)
            #     train_loss3.update(uncertain_loss.item(), 1)
            #
            #     loss_ce1_.update(loss_ce1.item(), 1)
            #     loss_con1_.update(loss_con1.item(), 1)
            #     loss_rad1_.update(loss_rad1.item(), 1)
            #     dice_loss1_.update(dice_loss1.item(), 1)
            #     train_loss1.update(supervised_loss.item(), 1)
            #
            #     loss_ce2_.update(loss_ce2.item(), 1)
            #     loss_con2_.update(loss_con2.item(), 1)
            #     loss_rad2_.update(loss_rad2.item(), 1)
            #     dice_loss2_.update(dice_loss2.item(), 1)
            #     train_loss2.update(certain_loss.item(), 1)
            #
            # logging.info('epoch : {}, '
            #              'lab_loss: {:.4f}, unlab_certain_loss: {:.4f}, unlab_uncertain_loss: {:.4f}, '
            #              'train_loss: {:.4f}, train_dice: {:.4f}, unlab_dice: {:.4f}, '
            #              'lab_rad: {:.4f}, lab_con: {:.4f}, unlab_rad: {:.4f}, unlab_con: {:.4f}'.format(
            #     epoch,
            #     train_loss1.val, train_loss2.val, train_loss3.val,
            #     train_loss.val, train_dice.val, unlab_dice.val,
            #     lab_rad_dice.val, lab_con_dice.val, unlab_rad_dice.val, unlab_con_dice.val))
            #
            # writer.add_scalar('supervised_loss/all', train_loss1.avg, epoch)
            # writer.add_scalar('supervised_loss/ce', loss_ce1_.avg, epoch)
            # writer.add_scalar('supervised_loss/rad', loss_rad1_.avg, epoch)
            # writer.add_scalar('supervised_loss/con', loss_con1_.avg, epoch)
            # writer.add_scalar('supervised_loss/dice', dice_loss1_.avg, epoch)
            #
            # writer.add_scalar('unsup_loss/certain_all', train_loss2.avg, epoch)
            # writer.add_scalar('unsup_loss/certain_ce', loss_ce2_.avg, epoch)
            # writer.add_scalar('unsup_loss/certain_rad', loss_rad2_.avg, epoch)
            # writer.add_scalar('unsup_loss/certain_con', loss_con2_.avg, epoch)
            # writer.add_scalar('unsup_loss/certain_dice', dice_loss2_.avg, epoch)
            #
            # writer.add_scalar('unsup_loss/uncertain_loss', train_loss3.avg, epoch)
            #
            # writer.add_scalar('acc/lab_dice', train_dice.avg, epoch)
            # writer.add_scalar('acc/unlab_dice', unlab_dice.avg, epoch)
            # writer.add_scalar('acc/unlab_rad_dice', unlab_rad_dice.avg, epoch)
            # writer.add_scalar('acc/unlab_con_dice', unlab_con_dice.avg, epoch)
            # writer.add_scalar('acc/lab_con_dice', lab_con_dice.avg, epoch)
            # writer.add_scalar('acc/lab_rad_dice', lab_rad_dice.avg, epoch)

            if (i + 1) % self.log_step == 0:
                self.info(f'Iter %d, global_iter: %d, supervised_loss: %.4f, certain_loss: %.4f, uncertain_loss: %f' %
                          (i, self.iter, supervised_loss.item(), certain_loss.item(), uncertain_loss))
            supervised_loss + certain_loss + uncertain_loss
            lr_ = cfg.lr * (1.0 - self.iter / (cfg.cora_epoch * cfg.num_iter_per_epoch)) ** 0.9

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
            self.iter += 1

    def pre_epoch(self, lb_loader, ul_loader, meter):
        train_loss, train_dice, cedc_loss_, loss_con_, loss_rad_ = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        self.net.train()

        lb_itr = iter(lb_loader)
        ul_itr = iter(ul_loader)

        for i in range(cfg.num_iter_per_epoch):
            bs = cfg.batch_size

            # get source data
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

            img = img.to(self.device, non_blocking=True)
            ul_img = img[bs:]
            msk = msk.to(self.device, non_blocking=True)
            m = mdl[0].item()

            # get output
            out = self.net(img)
            # [16,3,256,256]
            out_soft = torch.softmax(out, dim=1)
            out_back = out[:bs, 0, :, :].unsqueeze(dim=1)
            # print("out back is shape like :{}".format(out_back.shape))  # [8,1,256,256]
            """注意这里的输入"""
            out_0 = out[:bs, 1: 1 * cfg.n_label + 1, :, :]
            # print("out_back has shape {}".format(out_back.shape))
            # print("out_0 has shape {}".format(out_0.shape))
            out0 = torch.cat([out_back, out_0], dim=1)
            out_1 = out[:bs, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :]
            out1 = torch.cat([out_back, out_1], dim=1)
            out_2 = out[:bs, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :]
            out2 = torch.cat([out_back, out_2], dim=1)

            # supervised loss
            '''
            meanteacher:
            out[:bs] has shape torch.Size([8, 5, 256, 256])
            and min: -31.015661239624023 max:46.730770111083984
            msk has shape torch.Size([8, 256, 256])
            and min: 0 max:4
            coranet:
            out0 has shape torch.Size([8, 5, 256, 256])
            and min: -15.04464340209961 max:24.302494049072266
            msk has shape torch.Size([8, 256, 256])
            and min: 0 max:4
            '''
            cedc_loss = self.loss(out0, msk)
            # dice_loss = DiceLoss(out_soft[:bs, 0, :, :], msk)
            loss_con = self.conloss(out1, msk)
            loss_rad = self.radloss(out2, msk)
            loss = (cedc_loss + loss_con + loss_rad) / 4

            v, n = meter.collect_loss_by(loss.item(), m, img.size(0))
            meter.accumulate(v, n)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_ema_variable()

            # masks = self.get_mask(out0)
            masks = torch.argmax(out0, dim=1).cpu().numpy()
            lab = msk.cpu().numpy()
            train_dice.update(dc(masks, lab), 1)
            train_loss.update(loss.item(), 1)
            cedc_loss_.update(cedc_loss.item(), 1)
            # dice_loss_.update(dice_loss.item(), 1)
            loss_con_.update(loss_con.item(), 1)
            loss_rad_.update(loss_rad.item(), 1)

            if (i + 1) % self.log_step == 0:
                self.info(f'Iter %d, global_iter: %d, train_loss: %.4f, train_dice: %.4f'
                          f' cedc_loss: %.4f, loss_con: %.4f, loss_rad: %.4f' %
                          (i, self.iter, train_loss.avg, train_dice.avg, cedc_loss_.avg, loss_con_.avg, loss_rad_.avg))

            self.writer.add_scalar('pretrain/loss_ce', cedc_loss_.avg, self.iter)
            # self.writer.add_scalar('pretrain/loss_dice', dice_loss_.avg, self.iter)
            self.writer.add_scalar('pretrain/loss_con', loss_con_.avg, self.iter)
            self.writer.add_scalar('pretrain/loss_rad', loss_rad_.avg, self.iter)
            self.writer.add_scalar('pretrain/loss_all', train_loss.avg, self.iter)
            self.writer.add_scalar('pretrain/train_dice', train_dice.avg, self.iter)

            self.iter += 1

    def prefit(self, loader_type):  # 循环的训练模型
        tic = time.time()
        # Create data loaders and load npys for evaluation.
        if loader_type == 'inTurn':
            train_lb_loader = inlod.get_loader(cfg.base_root, 'train', self.fold, cfg.batch_size, cfg.data_aug)
            train_ul_loader = inlod.get_loader(cfg.base_root, 'val', self.fold, cfg.batch_size, cfg.data_aug)
            test_loader = inlod.get_loader(cfg.base_root, 'test', 0, cfg.batch_size)
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

        for epoch in range(cfg.pre_epoch):
            # train stage.
            train_meter.reset_cur()
            self.pre_epoch(train_lb_loader, train_ul_loader, train_meter)
            self.epoch += 1
            train_meter.update_cur()
            # train logs.
            lr = self.optimizer.param_groups[0]['lr']
            self.info('')
            self.info(f'lr: {lr}.')
            self.info(
                '[TRN] pre Epoch: %d(%d)/%d, elapsed: %.2fs,' % (epoch, best_epoch, cfg.pre_epoch, time.time() - tic)
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
            self.info('[TST] pre Epoch: %d/%d, elapsed: %.2fs,' % (epoch, cfg.pre_epoch, time.time() - tic)
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
                self.save_model(prefix='pre_best')
                self.save_ema_model(prefix='pre_ema_best')
                best_epoch = epoch

        self.save_model(prefix='pre_last')
        self.save_ema_model(prefix='pre_ema_last')

    def fit(self, loader_type):  # 循环的训练模型
        tic = time.time()
        # Create data loaders and load npys for evaluation.
        if loader_type == 'inTurn':
            train_lb_loader = inlod.get_loader(cfg.base_root, 'train', self.fold, cfg.batch_size, cfg.data_aug)
            train_ul_loader = inlod.get_loader(cfg.base_root, 'val', self.fold, 1, cfg.data_aug)
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

        # load and predict
        self.load_model(self.model_id, 'pre_best')
        self.load_ema_model(self.model_id, 'pre_ema_best')
        new_loader, plab_dice = self.pred_unlabel(train_ul_loader)
        self.writer.add_scalar('acc/plab_dice', plab_dice, 0)

        for epoch in range(cfg.cora_epoch):
            if epoch % cfg.pred_step == 0:
                new_loader, plab_dice = self.pred_unlabel(train_ul_loader)

            # train stage.
            train_meter.reset_cur()
            self.train_epoch(train_lb_loader, train_ul_loader, new_loader, train_meter)
            self.epoch += 1
            train_meter.update_cur()
            # train logs.
            lr = self.optimizer.param_groups[0]['lr']
            self.info('')
            self.info(f'lr: {lr}.')
            self.info('[TRN] Epoch: %d(%d)/%d, elapsed: %.2fs,' % (epoch, best_epoch, cfg.cora_epoch, time.time() - tic)
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
            self.info('[TST] Epoch: %d/%d, elapsed: %.2fs,' % (epoch, cfg.cora_epoch, time.time() - tic)
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

                out_back = out[:b, 0, :, :].unsqueeze(dim=1)
                # print("out back is shape like :{}".format(out_back.shape))  # [8,1,256,256]
                out_0 = out[:b, 1: 1 * cfg.n_label + 1, :, :]  # .unsqueeze(dim=1)
                out0 = torch.cat([out_back, out_0], dim=1)
                out_1 = out[:b, (1 * cfg.n_label + 1): 2 * cfg.n_label + 1, :, :]  # .unsqueeze(dim=1)
                out1 = torch.cat([out_back, out_1], dim=1)
                out_2 = out[:b, (2 * cfg.n_label + 1): 3 * cfg.n_label + 1, :, :]  # .unsqueeze(dim=1)
                out2 = torch.cat([out_back, out_2], dim=1)

                # supervised loss
                cedc_loss = self.loss(out0, msk)
                # dice_loss = DiceLoss(out_soft[:bs, 0, :, :], msk)
                loss_con = self.conloss(out1, msk)
                loss_rad = self.radloss(out2, msk)
                loss = (cedc_loss + loss_con + loss_rad) / 4

                if meter is not None:
                    v, n = meter.collect_loss_by(loss.item(), m, img.size(0))
                    meter.accumulate(v, n)
                pred = torch.argmax(out0, dim=1)
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

    trainer = coraNetTrainer(args.phase, args)
    if args.phase == 'train':
        # trainer.prefit('inTurn')
        trainer.fit('inTurn')  # firstly pretrain, then train
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
# using preds without softmax, 0.001, 0.0001, 62.78 epoch 17
