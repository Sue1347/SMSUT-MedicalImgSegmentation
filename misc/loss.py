# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceAndCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce=1., weight_dc=1., batch_dice=False):
        super(DiceAndCrossEntropyLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dc = weight_dc
        self.ce = nn.CrossEntropyLoss()
        self.dc = SoftDiceLoss(batch_dice=batch_dice)

    def forward(self, x, y):
        dc_loss = self.dc(x, y) if self.weight_dc != 0 else 0.
        ce_loss = self.ce(x, y) if self.weight_ce != 0 else 0.
        loss = self.weight_dc * dc_loss + self.weight_ce * ce_loss
        return loss


def get_tp_fp_fn_tn(output, gt, dims=(2, 3)):
    x_shape = output.shape
    with torch.no_grad():
        gt_onehot = torch.zeros(x_shape)
        if output.device.type == 'cuda':
            gt_onehot = gt_onehot.cuda(output.device.index)
            gt = gt.cuda(output.device.index)
        gt_onehot.scatter_(1, gt.unsqueeze(1), 1)

    tp = torch.sum(output * gt_onehot, dim=dims)
    fp = torch.sum(output * (1. - gt_onehot), dim=dims)
    fn = torch.sum((1. - output) * gt_onehot, dim=dims)
    tn = torch.sum((1. - output) * (1. - gt_onehot), dim=dims)
    return tp, fp, fn, tn  # Size(C or B, C)


class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=False, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, x, y):
        """
        :param x: torch.FloatArray, Size(B, BG + C, H, W)
        :param y: torch.LongArray, Size(B, H, W)
        :return:
        """
        x = F.softmax(x, dim=1)
        dims = (0, 2, 3) if self.batch_dice else (2, 3)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, dims)
        inter = 2 * tp + self.smooth
        union = 2 * tp + fp + fn + self.smooth
        dc = inter / (union + 1e-8)
        # print(dc.shape)
        if self.batch_dice:
            dc = dc[1:]  # ignore background
        else:
            dc = dc[:, 1:]
        return 1. - dc.mean()


if __name__ == '__main__':
    import config as cfg
    from data_loader.baseLoader import get_loader
    loader = get_loader(cfg.png_root, phase='train', modal='all', batch_size=8, load_in_ram=False)
    itr = iter(loader)
    img, lbl, mdl, inm = next(itr)
    output = torch.randn(8, 5, 256, 256)
    print(img.shape, lbl.shape, output.shape)
    print(mdl.shape, len(mdl.shape))
    exit()

    # loss1 = SoftDiceLoss(reduce_batch=True)
    # loss2 = SoftDiceLoss(reduce_batch=False)
    loss1 = DiceAndCrossEntropyLoss(reduce_batch=True)
    loss2 = DiceAndCrossEntropyLoss(reduce_batch=False)
    l1 = loss1(output, lbl)
    l2 = loss2(output, lbl)
    print(l1, l2.mean(), l2)
