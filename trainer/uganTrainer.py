# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())
import random
import numpy as np
import argparse
import time
import shutil
from os.path import join as pjoin
from medpy.metric import dc 
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torchvision.utils import save_image

from trainer.baseTrainer import BaseTrainer
from network.ugan import UGAN, Discriminator
import config as cfg
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml
from misc.visualize import count_param_number
from misc.select_used_modules import make_closure_fast
from misc.loss import DiceAndCrossEntropyLoss
from data_loader import balanceLoader, baseLoader, inTurnLoader


class UGANTrainer(BaseTrainer):
    def __init__(self, phase, args):
        # Hyper params.
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10
        self.lambda_seg = 10
        self.lambda_shp = 10
        self.lambda_shp_lazy = 20

        self.log_step = 50
        self.n_critic = 1

        self.beta1 = 0.9
        self.beta2 = 0.999

        super(UGANTrainer, self).__init__(phase, args)

    def build_network(self):
        self.net = UGAN(cfg.img_channels, cfg.n_label+1, cfg.n_modal, cfg.base_width)
        count_param_number(self.net, verbose=True, logger=self)
        self.net.to(self.device)

        self.D = Discriminator(cfg.input_size, cfg.n_modal, cfg.base_width, max_width=256 if cfg.base_width==16 else 512)
        count_param_number(self.D, verbose=True, logger=self)
        self.D.to(self.device)

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            self.D = nn.DataParallel(self.D)

        if self.phase == 'train':
            beta1, beta2 = self.beta1, self.beta2
            self.optimizer = SGD(self.net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            self.d_optimizer = Adam(self.D.parameters(), cfg.lr, [beta1, beta2], weight_decay=cfg.weight_decay)

    def load_model(self, model_idx, which_ckpt):
        G_path = pjoin(self.expr_root, model_idx, 'ckpt', f'{which_ckpt}_G.ckpt')
        D_path = pjoin(self.expr_root, model_idx, 'ckpt', f'{which_ckpt}_D.ckpt')
        self.net.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print(f'[*] Load G and D from {G_path}.')

    def save_model(self, prefix):
        assert self.phase == 'train'
        G_path = pjoin(self.expr_root, self.model_idx, 'ckpt', f'{prefix}_G.ckpt')
        D_path = pjoin(self.expr_root, self.model_idx, 'ckpt', f'{prefix}_D.ckpt')
        if isinstance(self.net, nn.DataParallel):
            torch.save(self.net.module.state_dict(), G_path)
            torch.save(self.D.module.state_dict(), D_path)
        else:
            torch.save(self.net.state_dict(), G_path)
            torch.save(self.D.state_dict(), D_path)
        print(f'[*] Save G and D to {G_path}.')

    def label2onehot(self, modals, dim=cfg.n_modal):
        batch_size = modals.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), modals.long()] = 1
        return out
    
    def create_vectors(self, vec_org, dim):
        vec_trg_list = []
        for i in range(dim):
            vec_trg = self.label2onehot(torch.ones(vec_org.size(0)) * i, dim)
            vec_trg_list.append(vec_trg.to(self.device))
        return vec_trg_list

    @staticmethod
    def denorm(x):
        out = (x + 1.) / 2.
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight,
                                   retain_graph=True, create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def train_epoch(self, lb_loader, ul_loader, meter):
        self.net.train()
        self.D.train()
        
        # Hyper parameters.
        lambda_cls, lambda_gp, lambda_gp = self.lambda_cls, self.lambda_gp, self.lambda_gp
        lambda_seg, lambda_rec = self.lambda_seg, self.lambda_rec
        lambda_shp = self.epoch * (self.lambda_shp / self.lambda_shp_lazy)
        lambda_shp = min(lambda_shp, lambda_seg)
        n_critic = self.n_critic
        print(f'\nlambda_seg: {lambda_seg}, lambda_shp: {lambda_shp}.')

        itr = iter(lb_loader)
        tic = time.time()

        # fixed images for debug.
        x_fixed, _, modal_org, inm = next(itr)
        print(inm)
        vec_fixed_org = self.label2onehot(modal_org, cfg.n_modal)
        x_fixed = x_fixed.to(self.device, non_blocking=True)
        vec_fixed_org = vec_fixed_org.to(self.device)
        vec_fixed_list = self.create_vectors(vec_fixed_org, cfg.n_modal)
        for i in range(n_critic * cfg.num_iter_per_epoch):
            try:
                x_real, y_real, modal_org, _ = next(itr)
            except StopIteration:
                itr = iter(lb_loader)
                x_real, y_real, modal_org, _ = next(itr)
            m = modal_org[0].item()

            mj = random.randint(0, cfg.n_modal-1)
            modal_trg = torch.zeros_like(modal_org).fill_(mj)

            vec_org = self.label2onehot(modal_org, cfg.n_modal)
            vec_trg = self.label2onehot(modal_trg, cfg.n_modal)

            x_real = x_real.to(self.device, non_blocking=True)
            y_real = y_real.to(self.device, non_blocking=True)
            vec_org = vec_org.to(self.device)
            vec_trg = vec_trg.to(self.device)
            vec_ot = vec_trg - vec_org
            vec_to = vec_org - vec_trg
            modal_org = modal_org.to(self.device)
            modal_trg = modal_trg.to(self.device)

            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = F.cross_entropy(out_cls, modal_org)

            _, x_fake = self.net(x_real, vec_ot)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            alpha = torch.randn(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
            self.d_optimizer.zero_grad(); self.optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            loss = dict(D_real=d_loss_real.item(), D_fake=d_loss_fake.item(), 
                        D_cls=d_loss_cls.item(), D_gp=d_loss_gp.item())
            if (i + 1) % n_critic == 0:
                y_fake, x_fake = self.net(x_real, vec_ot)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = F.cross_entropy(out_cls, modal_trg)
                g_loss_seg = self.loss(y_fake, y_real)
                v, n = meter.collect_loss_by(g_loss_seg.item(), m, x_real.size(0))
                meter.accumulate(v, n)

                y_rec, x_rec = self.net(x_fake, vec_to)
                g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
                g_loss_shp = self.loss(y_rec, y_real)

                g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls + \
                    lambda_seg * g_loss_seg + lambda_shp * g_loss_shp
                self.d_optimizer.zero_grad(); self.optimizer.zero_grad()
                g_loss.backward()
                self.optimizer.step()

                loss['G_fake'] = g_loss_fake.item()
                loss['G_rec'] = g_loss_rec.item()
                loss['G_cls'] = g_loss_cls.item()
                loss['G_seg'] = g_loss_seg.item()
                loss['G_shp'] = g_loss_shp.item()

            if (i + 1) % (n_critic * self.log_step) == 0:
                log = 'Iter: %d/%d(%d), elapsed: %.2fs,' % \
                    (i, n_critic * cfg.num_iter_per_epoch, self.iter, time.time() - tic)
                tic = time.time()
                for k, v in loss.items():
                    log += ' %s: %.4f,' % (k, v)
                print(log, flush=True)

            lr_ = cfg.lr * (1.0 - self.iter / (cfg.max_epoch * cfg.num_iter_per_epoch)) ** 0.9 
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = lr_
            self.iter += 1

        with torch.no_grad():
            x_fake_list = [x_fixed]
            for vec_fixed in vec_fixed_list:
                vec_ot = vec_fixed - vec_fixed_org
                _, x_fake = self.net(x_fixed, vec_ot)
                x_fake_list.append(x_fake)
            x_concat = torch.cat(x_fake_list, dim=3)
            sample_path = pjoin(self.expr_root, self.model_idx, 'sample', f'train-{self.epoch+1}-images.jpg')
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            print(f'[*] Saved real and fake images into {sample_path}.')

    def validate_epoch(self, loader, npys, meter=None, save_path=None):
        self.net.eval() 
        self.D.eval()
        prd_npys, n_prd_slic = dict(), 0
        for k, v in npys.items():
            prd_npys[k] = np.zeros(v.shape, dtype=v.dtype)
        with torch.no_grad():
            for x_real, y_real, mdl, inm in loader:
                b, c, h, w = x_real.shape 
                if b != cfg.batch_size:
                    pad = torch.zeros((cfg.batch_size - b, c, h, w), dtype=x_real.dtype)
                    x_real = torch.cat([x_real, pad], dim=0)
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                m = mdl[0].item()

                y_fake, x_fake = self.net(x_real)
                if b != cfg.batch_size:
                    y_fake, x_fake = y_fake[:b], x_fake[:b]
                sample_loss = self.loss(y_fake, y_real)
                if meter is not None:
                    v, n = meter.collect_loss_by(sample_loss.item(), m, x_real.size(0))
                    meter.accumulate(v, n)
                pred = torch.argmax(y_fake, dim=1)
                pred = pred.detach().cpu().numpy()

                for i in range(b):
                    if save_path is not None:
                        slic = pred[i]
                        slic = Image.fromarray(slic.astype(np.uint8))
                        save_p = pjoin(save_path, f'{inm[i]}.png')
                        slic.save(save_p)
                    m, pid, z = inm[i].split('_')
                    prd_npys[f'{m}_{pid}'][int(z)] = pred[i]
                    n_prd_slic += 1
        return n_prd_slic, prd_npys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', type=str, choices=('train', 'test'))
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-nm', '--expr_name', type=str)

    parser.add_argument('-i', '--model_id', type=str, help='only for test')
    parser.add_argument('-wh', '--which_ckpt', type=str, default='last')
    args = parser.parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    trainer = UGANTrainer(args.phase, args)
    if args.phase == 'train':
        trainer.fit('inTurn')
    else:
        trainer.load_model(args.model_id, args.which_ckpt)
        expr_root = pjoin(trainer.expr_root, args.model_id)
        trainer.test('inTurn', expr_root)
