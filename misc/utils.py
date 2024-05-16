# -*- coding: utf-8 -*-

import os
import numpy as np
from copy import deepcopy
import yaml
from collections import OrderedDict
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from medpy.metric import dc, hd, asd, assd
from skimage import measure

import config as cfg
from misc.loss import get_tp_fp_fn_tn


def connected_components(pred):
    predict = np.zeros_like(pred)
    for i in range(cfg.n_modal):
        pre = (pred == i+1).astype(np.uint8)
        labels = measure.label(pre, connectivity=2)
        retain_num = []
        ratio = 0.1
        thresold = ratio * np.sum(labels != 0)
        for j in range(1, np.max(labels) + 1):
            if np.sum(labels == j) > thresold:
                retain_num.append(j)
            else:
                labels[labels == j] = 0
        labels[labels != 0] = 1
        labels = np.uint8(labels)
        predict += labels*(i+1)

    # print(type(labels))
    return np.uint8(predict)


def maybe_mkdir(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.mkdir(arg)
    return


def read_yaml(path):
    with open(path, 'r') as f:
        f = yaml.load(f, Loader=yaml.FullLoader)
    return f


def write_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)
    return


class Meter(object):
    def __init__(self, min_better_keys: list, max_better_keys: list, alpha: float = 1.):
        self.configs = OrderedDict()
        self.alpha = alpha
        self.__register(min_better_keys, max_better_keys)
        self.best_values = self.get_empty_dict()
        self.pre_values = None
        self.cur_values = self.get_empty_dict()
        self.n = self.get_empty_dict()

    def get_empty_dict(self):
        temp = {k: 0 for k in self.configs.keys()}
        return temp

    def __register(self, min_better_keys, max_better_keys):
        for k in min_better_keys:
            self.configs[k] = 'min'
        for k in max_better_keys:
            self.configs[k] = 'max'

    def accumulate(self, values: dict, n: dict):
        for k, v in values.items():
            self.cur_values[k] += v
            self.n[k] += n[k]

    def update_cur(self, reset_best=False):
        for k in self.configs.keys():
            if self.n[k] != 0:
                self.cur_values[k] /= self.n[k]
            if self.pre_values is not None:
                self.cur_values[k] = (1. - self.alpha) * self.pre_values[k] + self.alpha * self.cur_values[k]

        if self.pre_values is None or reset_best:
            self.best_values = deepcopy(self.cur_values)
            self.pre_values = deepcopy(self.cur_values)
        else:
            for k, f in self.configs.items():
                if f == 'min' and self.cur_values[k] < self.best_values[k]:
                    self.best_values[k] = self.cur_values[k]
                elif f == 'max' and self.cur_values[k] > self.best_values[k]:
                    self.best_values[k] = self.cur_values[k]
                self.pre_values[k] = self.cur_values[k]

    def reset_cur(self):
        self.cur_values = self.get_empty_dict()
        self.n = self.get_empty_dict()

    @staticmethod
    def collect_loss_by(sample_loss, modal_id, n):
        """
        :param sample_loss: 
        :param modal_idx: 
        :param n_modal: int
        :return:
        """
        k = 'loss_' + str(modal_id)
        a = {'loss': sample_loss * n, k: sample_loss * n}
        b = {'loss': n, k: n}
        return a, b

    @staticmethod
    def collect_dice_by(output, gt, modal_idxs, n_modal, smooth=1e-5):
        """
        :param output: torch.cuda.FloatArray, Size(B, C, H, W)
        :param gt: torch.cuda.LongArray, Size(B, H, W)
        :param modal_idxs: torch.LongArray, Size(B)
        :param n_modal: int
        :param smooth: float
        :return:
        """
        y_onehot = torch.zeros(output.shape)
        y_onehot = y_onehot.cuda(output.device.index)
        y = torch.argmax(output, dim=1, keepdim=True)  # Size(B, 1, H, W)
        y_onehot.scatter_(1, y, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(y_onehot.float(), gt)
        inter = 2 * tp + smooth
        union = 2 * tp + fp + fn + smooth
        sample_dices = inter / union  # Size(B, C)
        _, c = sample_dices.shape
        sample_dices = torch.sum(sample_dices[:, 1:], dim=1) / (c - 1)  # Size(B), ignore background.

        dice = [0 for _ in range(n_modal)]
        n = [0 for _ in range(n_modal)]
        for sd, mi in zip(sample_dices, modal_idxs):
            i = mi.item()
            dice[i] += sd.item()
            n[i] += 1
        a = {f'dice_{i}': dice[i] for i in range(n_modal)}
        a['dice'] = sum(dice)
        b = {f'dice_{i}': n[i] for i in range(n_modal)}
        b['dice'] = sum(n)
        return a, b

    def __repr__(self):
        format_string = ''
        for k in self.configs.keys():
            if '_' in k:
                typ, m = k.split('_')
                new_k = f'{typ}_{cfg.Modality(int(m)).name}'
            else:
                new_k = k
            format_string += ' %s: %.4f/%.4f,' % (new_k, self.cur_values[k], self.best_values[k])
        return format_string


def get_label_npys(png_root, modal, phase):
    retn, n = {}, 0
    split = read_yaml(pjoin(png_root, cfg.split_yaml))
    if modal == 'all':
        modal = cfg.Modality.__members__
    else:
        modal = [modal]

    for m in modal:
        for p in split[m][phase]:
            npy_path = pjoin(png_root, m, p, f'{m}_{p}.npy')
            npy = np.load(npy_path)
            n += npy.shape[0]
            retn[f'{m}_{p}'] = npy
    return n, retn


def get_mo_matrix(prd_npys, gt_npys):
    """ Modality-Organ Dice Score Matrix.
    """
    matrix = np.zeros((cfg.n_modal, cfg.n_label))
    n = np.zeros((cfg.n_modal, 1))
    for k in gt_npys.keys():
        m, _ = k.split('_')
        m = cfg.Modality[m].value

        p, g = prd_npys[k], gt_npys[k]
        for i in range(cfg.n_label):
            j = i + 1
            s = dc((p == j).astype(np.int), (g == j).astype(np.int))
            matrix[m][i] += s
        n[m] += 1
    n[n == 0] += 1e-8  # prevent dividing zero.
    matrix /= n

    temp_matrix = np.zeros((cfg.n_modal + 1, cfg.n_label + 1))
    temp_matrix[:cfg.n_modal, :cfg.n_label] = matrix
    matrix = temp_matrix
    matrix[-1, :] = np.mean(matrix[0: cfg.n_modal], axis=0)
    matrix[:, -1] = np.mean(matrix[:, 0: cfg.n_label], axis=1)
    return matrix


def get_all_matrix(prd_npys, gt_npys):
    """ Modality-Organ
    Dice Score Matrix
    Hausdorff Distance
    Average Symmetric Surface Distance
    """
    matrix = np.zeros((cfg.n_modal, cfg.n_label))
    hd_matrix = np.zeros((cfg.n_modal, cfg.n_label))
    assd_matrix = np.zeros((cfg.n_modal, cfg.n_label))
    n = np.zeros((cfg.n_modal, 1))
    dc_res = ["dice_results", "0"]
    for k in gt_npys.keys():
        dc_results = 0
        m, idx = k.split('_')
        m = cfg.Modality[m].value

        p, g = prd_npys[k], gt_npys[k]

        p1 = connected_components(p)  # 将预测变得更连通，更平滑

        for i in range(p1.shape[0]):
            temp = connected_components(p1[i, :, :])
            p1[i, :, :] = temp

        # print("******************get all matrix: prd k is shape like :*****************")
        # print("{} {} {}".format(p.shape[0],p.shape[1],p.shape[2]))//24(slices),256,256
        maxassd = 0
        for i in range(cfg.n_label):
            j = i + 1
            predx = (p1 == j).astype(np.bool_).astype(np.int)
            gx = (g == j).astype(np.bool_).astype(np.int)
            s = dc(predx, gx)
            # print("**************** p1 == j as type bool")
            # print("******** min: {}  max: {}".format(predx.min(), predx.max()))
            # print("**************** g == j as type bool:")
            # print("******** min: {}  max: {}".format(gx.min(), gx.max()))
            if predx.max() == 0:
                r = maxassd
            else: r = assd(predx, gx)
            t = s  # hd(predx, gx)
            maxassd = maxassd if maxassd > r else r
            matrix[m][i] += s
            hd_matrix[m][i] += t
            assd_matrix[m][i] += r
            dc_results += s
        n[m] += 1
        dc_res=np.append(dc_res, [k,str((dc_results/cfg.n_label))], axis = 0)
    n[n == 0] += 1e-8  # prevent dividing zero.
    matrix /= n
    hd_matrix /= n
    assd_matrix /= n

    '''
    placing the average into matrix.
    '''
    temp_matrix = np.zeros((cfg.n_modal + 1, cfg.n_label + 1))
    temp_matrix[:cfg.n_modal, :cfg.n_label] = matrix
    matrix = temp_matrix
    matrix[-1, :] = np.mean(matrix[0: cfg.n_modal], axis=0)
    matrix[:, -1] = np.mean(matrix[:, 0: cfg.n_label], axis=1)

    temp_matrix = np.zeros((cfg.n_modal + 1, cfg.n_label + 1))
    temp_matrix[:cfg.n_modal, :cfg.n_label] = hd_matrix
    hd_matrix = temp_matrix
    hd_matrix[-1, :] = np.mean(hd_matrix[0: cfg.n_modal], axis=0)
    hd_matrix[:, -1] = np.mean(hd_matrix[:, 0: cfg.n_label], axis=1)

    temp_matrix = np.zeros((cfg.n_modal + 1, cfg.n_label + 1))
    temp_matrix[:cfg.n_modal, :cfg.n_label] = assd_matrix
    assd_matrix = temp_matrix
    assd_matrix[-1, :] = np.mean(assd_matrix[0: cfg.n_modal], axis=0)
    assd_matrix[:, -1] = np.mean(assd_matrix[:, 0: cfg.n_label], axis=1)

    # print(dc_res)

    # np.savetxt(pjoin(cfg.expr_root, 'prd_list.csv'),  dc_res, fmt="%s", delimiter=",")
    
    return matrix, hd_matrix, assd_matrix


if __name__ == '__main__':
    from data_loader.baseLoader import get_loader

    loader = get_loader(cfg.png_root, phase='train', modal='all', batch_size=8, load_in_ram=False)
    itr = iter(loader)
    img, lbl, mdl, inm = next(itr)
    meter = Meter([], [])
    output = torch.randn(8, 5, 256, 256)
    a, b = meter.collect_dice_by(output, lbl, mdl, 4)
    print(a, b)
