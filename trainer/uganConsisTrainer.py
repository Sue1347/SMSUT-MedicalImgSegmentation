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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter # pip install protobuf==3.20.1 
import torch.nn.functional as F 
from torchvision.utils import save_image

from trainer.uganShp0Trainer import UGANShp0Trainer
from network.ugan import UGAN, Discriminator
import config as cfg
from misc.utils import maybe_mkdir, Meter, get_label_npys, write_yaml
from misc.visualize import count_param_number
from misc.select_used_modules import make_closure_fast
from misc.loss import DiceAndCrossEntropyLoss
from data_loader import balanceLoader, baseLoader, inTurnLoader

import warnings
warnings.filterwarnings("ignore")

class UGANConsisTrainer(UGANShp0Trainer):
    def __init__(self, phase, args):
        super(UGANConsisTrainer, self).__init__(phase, args)

        self.lambda_semi = 10

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        # self.consis_loss = nn.MSELoss()

    def consistency_loss(self, source, target):
        # _, _, h, w = source.shape
        # kl1 = self.kl_div(F.log_softmax(source, dim=1), F.softmax(target, dim=1))
        # kl2 = self.kl_div(F.log_softmax(target, dim=1), F.softmax(source, dim=1))
        # loss = 0.5 * (kl1 + kl2)
        # return kl1 / (h * w)
        # return self.consis_loss(F.softmax(source, dim=1), F.softmax(target, dim=1))
        target = torch.argmax(target, dim=1)
        return self.loss(source, target)

    def nce_loss(self, feat_x_pool, feat_f_pool):
        n_layers = cfg.nce_layers# because we have 128 channels, so I may change them into [0,15,31,63,127]
        n = len(n_layers)

        total_nce_loss = 0.0
        for f_f, f_x, crit, nce_layer in zip(feat_f_pool, feat_x_pool, self.criterionNCE, n_layers):
            loss = crit(f_f, f_x) * 1.0
            total_nce_loss += loss.mean()

        return total_nce_loss / n

    def train_epoch(self, lb_loader, ul_loader, meter):
        self.net.train()
        # self.netF.train()
        self.D.train()
        # Hyper parameters.
        lambda_cls, lambda_gp = self.lambda_cls, self.lambda_gp
        lambda_seg, lambda_rec = self.lambda_seg, self.lambda_rec

        lambda_semi = self.lambda_semi * self.sigmoid_rampup(self.epoch, cfg.max_epoch)

        n_critic = self.n_critic
        print(f'\nlambda_seg: {lambda_seg}, lambda_semi: {lambda_semi}.')

        lb_itr = iter(lb_loader)
        ul_itr = iter(ul_loader)
        tic = time.time()

        # fixed images for debug.
        x_fixed1, _, modal_org1, inm1 = next(lb_itr)
        x_fixed2, _, modal_org2, inm2 = next(ul_itr)
        x_fixed = torch.cat([x_fixed1, x_fixed2], dim=0)
        modal_org = torch.cat([modal_org1, modal_org2], dim=0)
        # # assert torch.all(modal_org1 == modal_org2), f'{modal_org1}, {modal_org2}'
        self.info(inm1 + inm2)
        vec_fixed_org = self.label2onehot(modal_org, cfg.n_modal)
        x_fixed = x_fixed.to(self.device, non_blocking=True)
        vec_fixed_org = vec_fixed_org.to(self.device)
        vec_fixed_list = self.create_vectors(vec_fixed_org, cfg.n_modal)

        for i in range(n_critic * cfg.num_iter_per_epoch):
            bs = cfg.batch_size

            try:
                x_real1, y_real, modal_org1, _ = next(lb_itr)
            except StopIteration:
                lb_itr = iter(lb_loader) 
                x_real1, y_real, modal_org1, _ = next(lb_itr)

            try:
                x_real2, _, modal_org2, _ = next(ul_itr)
            except StopIteration:
                ul_itr = iter(ul_loader)
                x_real2, _, modal_org2, _ = next(ul_itr)
            
            x_real = torch.cat([x_real1, x_real2], dim=0)
            modal_org = torch.cat([modal_org1, modal_org2], dim=0)
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

            _, x_fake, feat_x_pool, sample_ids = self.net(x_real, vec_ot)
            # feat_x_pool = self.netF(tsl_out)
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
                y_fake, x_fake, feat_x_pool, sample_ids = self.net(x_real, vec_ot)
                # feat_x_pool = self.netF(tsl_out)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = F.cross_entropy(out_cls, modal_trg)
                g_loss_seg = self.loss(y_fake[:bs], y_real)
                v, n = meter.collect_loss_by(g_loss_seg.item(), m, bs)
                meter.accumulate(v, n)

                y_rec, x_rec, feat_f_pool, _ = self.net(x_fake, vec_to, sample_ids=sample_ids)
                # feat_f_pool = self.netF(tsl_out)
                g_loss_rec = torch.mean(torch.abs(x_real - x_rec))

                # Semi supervised.
                if self.iter < 1000:
                    g_loss_semi = torch.tensor(0., device=self.device)
                else:
                    g_loss_semi = self.consistency_loss(y_rec, y_fake)

                # nce loss
                # self.criterionNCE = PatchNCELoss(cfg.batch_size).to(self.device)
                g_loss_nce = self.nce_loss(feat_x_pool, feat_f_pool)

                g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls + \
                    lambda_seg * g_loss_seg + \
                    lambda_semi * g_loss_semi +\
                    1.0 * g_loss_nce
                self.d_optimizer.zero_grad(); self.optimizer.zero_grad()
                g_loss.backward()
                self.optimizer.step()


                loss['G_fake'] = g_loss_fake.item()
                loss['G_rec'] = g_loss_rec.item()
                loss['G_cls'] = g_loss_cls.item()
                loss['G_seg'] = g_loss_seg.item()
                loss['G_semi'] = g_loss_semi.item()
                loss['G_nce'] = g_loss_nce.item()

            if (i + 1) % (n_critic * self.log_step) == 0:
                log = 'Iter: %d/%d(%d), elapsed: %.2fs,' \
                    % (i, n_critic * cfg.num_iter_per_epoch, self.iter, time.time() - tic)
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
                _, x_fake, _, _ = self.net(x_fixed, vec_ot)
                x_fake_list.append(x_fake)
            x_concat = torch.cat(x_fake_list, dim=3)
            sample_path = pjoin(self.expr_root, self.model_idx, 'sample', f'train-{self.epoch+1}-images.jpg')
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            print(f'[*] Saved real and fake images into {sample_path}.')

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
                # print(inm) # ['ct_027_000', 'ct_027_001', 'ct_027_002', 'ct_027_003', 'ct_027_004', 'ct_027_005', 'ct_027_006', 'ct_027_007']
                # exit()
                b, c, h, w = img.shape
                count += b 
                img = img.to(self.device, non_blocking=True)

                vec_fixed_org = self.label2onehot(mdl, cfg.n_modal)
                vec_fixed_org = vec_fixed_org.to(self.device)
                vec_fixed_list = self.create_vectors(vec_fixed_org, cfg.n_modal)
                x_fake_list = [img]
                for vec_fixed in vec_fixed_list:
                    vec_ot = vec_fixed - vec_fixed_org
                    _, x_fake, _, _ = self.net(img, vec_ot)
                    x_fake_list.append(x_fake)
                img_fake = torch.cat(x_fake_list, dim=3)
                # sample_path = save_a = pjoin(pred_root, inm[i] + 'ori.jpg')
                # pjoin(self.expr_root, self.model_idx, 'sample', f'train-{self.epoch + 1}-images.jpg')
                # save_image(self.denorm(img_fake.data.cpu()), sample_path, nrow=1, padding=0)
                # print(f'[*] Saved real and fake images into {pred_root}.')

                out, _, _, _ = self.net(img)
                pred = torch.argmax(out, dim=1)
                pred = pred.detach().cpu().numpy()
                # img_fake = img_fake.squeeze().cpu().numpy()
                img = img.squeeze().cpu().numpy()
                msk = msk.cpu().numpy()
                # print(" The size of pred: {}".format(pred.shape))
                # print(" The size of img: {}".format(img.shape))
                # print(" The size of img_fake: {}".format(img_fake.shape))

                for i in range(b):
                    p, m = pred[i], msk[i]
                    a = img[i]#, img_fake[i]
                    p, m = colorize(p), colorize(m)
                    # print(" pred min: {} pred max: {}".format(p.astype(np.uint8).min(), p.astype(np.uint8).max()))
                    # print(" img_fake min: {} img_fake max: {}".format(b.min(), b.max()))
                    a = (a+1)*255
                    # b = np.multiply(np.add(b,1), 255)
                    # print(a.shape)
                    p = Image.fromarray(p.astype(np.uint8))
                    m = Image.fromarray(m.astype(np.uint8))
                    a = Image.fromarray(a.astype(np.uint8))
                    # b = Image.fromarray(b).convert('RGB')
                    
                    fk = (img_fake[i].data.cpu() + 1.)/2 * 255
                    # print(fk.squeeze(0).shape)
                    b = Image.fromarray(fk.squeeze(0).numpy().astype(np.uint8))
                    # print(fk.min(),fk.max())
                    
                    mod, pid, z = inm[i].split('_')
                    if mod+'_'+pid in ['ct_028', 't1in_037','t1out_015', 't2_032']: # selected visualization samples
                        save_p = pjoin(pred_root, inm[i] + 'pse.jpg')
                        p.save(save_p)
                        save_m = pjoin(pred_root, inm[i] + 'gt.jpg')
                        m.save(save_m)
                        save_a = pjoin(pred_root, inm[i] + 'ori.jpg')
                        a.save(save_a)
                        save_b = pjoin(pred_root, inm[i] + 'fk.jpg')
                        # save_image(self.denorm(b), save_b, nrow=1, padding=0)
                        # b.save(save_b)
                        # save_image(self.denorm(img_fake[i].data.cpu()), save_b, nrow=1, padding=0)
                        # exit()
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

    trainer = UGANConsisTrainer(args.phase, args)
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
