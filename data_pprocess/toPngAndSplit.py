# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import SimpleITK as sitk
import yaml
from os.path import join as pjoin
import random
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

from misc.utils import read_yaml, write_yaml, maybe_mkdir

import config


def to_png(src, dst, debug=False):
    maybe_mkdir(dst)
    for modality in os.listdir(src):
        print(modality)
        dst_modal_root = pjoin(dst, modality)
        maybe_mkdir(dst_modal_root)

        src_modal_img_root = pjoin(src, modality, 'images')
        for pid in tqdm(os.listdir(src_modal_img_root)):
            src_img_path = pjoin(src_modal_img_root, pid)
            image = sitk.GetArrayFromImage(sitk.ReadImage(src_img_path))
            label = sitk.GetArrayFromImage(sitk.ReadImage(src_img_path.replace('images', 'labels')))

            if modality == 'ct':
                mi, ma = -1000, 400
            else:
                mi, ma = np.percentile(image, 0.05), np.percentile(image, 99.5)
            image[image < mi] = mi
            image[image > ma] = ma
            image = (image - image.min()) / (image.max() - image.min()) * 255

            dst_modal_pid_root = pjoin(dst_modal_root, pid.split('_')[1][:3])
            dst_modal_pid_img_root = pjoin(dst_modal_pid_root, 'images')
            dst_modal_pid_lbl_root = pjoin(dst_modal_pid_root, 'labels')
            maybe_mkdir(dst_modal_pid_root, dst_modal_pid_img_root, dst_modal_pid_lbl_root)
            if debug:
                dst_modal_pid_vis_root = pjoin(dst_modal_pid_root, 'vis')
                maybe_mkdir(dst_modal_pid_vis_root)
                n_label = np.max(label)
            np.save(pjoin(dst_modal_pid_root, pid.replace('.nii.gz', '.npy')), label)  # 3d array for evaluation.

            for z in range(image.shape[0]):
                img, lbl, vis = image[z], label[z], label[z]
                img = Image.fromarray(img.astype(np.uint8))
                lbl = Image.fromarray(lbl.astype(np.uint8))

                img_name = f'{modality}_{pid.split("_")[1][:3]}_{str(z).rjust(3, "0")}.png'
                img_save_path = pjoin(dst_modal_pid_img_root, img_name)
                lbl_save_path = pjoin(dst_modal_pid_lbl_root, img_name)
                img.save(img_save_path)
                lbl.save(lbl_save_path)

                if debug:
                    d = 255 // n_label
                    for i in range(1, n_label+1):
                        vis[vis == i] = d * i
                    vis = Image.fromarray(vis.astype(np.uint8))
                    vis.save(pjoin(dst_modal_pid_vis_root, img_name))
        # break
    return


def split_train_val_test(data_root, save_root=''):
    ratios = (1, 9, 10)
    # ratios = (1, 4, 5) (1, 9, 10) (3,7,10) (4,6,10) (5, 5, 10)
    # saml: 145 195 375 465 555
    idx = ratios[2]

    modality2volume = dict()
    for modality in config.mod_type: # os.listdir(data_root)
        print(modality)
        if modality.endswith('.yaml'):
            continue
        volumes = list(os.listdir(pjoin(data_root, modality)))
        modality2volume[modality] = volumes

    split = dict()
    for modality, volumes in modality2volume.items():
        n_volume = len(volumes)
        n = len(volumes) // sum(ratios)
        n_train = int(ratios[0] / sum(ratios) * n_volume)
        n_val = int(ratios[1] / sum(ratios) * n_volume)
        # n_train = 1
        # n_val = int((ratios[0] + ratios[1]) / sum(ratios) * n_volume) - 1
        if n_train == 0:
            n_train = 1
            n_val = n_val-1

        if n_val == 0:
            n_val = 1
            n_train = n_train-1

        indexs = np.random.permutation(np.arange(n_volume))
        print('n volumw is  {} n is {}'.format(n_volume, n))
        print('n train is  {} n_val is {}'.format(n_train, n_val))
        print("****** indexs is {}".format(indexs))

        trains, vals = [], []
        for i in range(5):  # just the num of random data arrays
            end = n_val + n_train
            if end >= (i*n+n_train):
                train = indexs[i*n: min(end, i*n+n_train)]
                # print("train is from {} to {}".format(i*n, min(end, i*n+n_train)))
                val = np.concatenate((indexs[0:i * n], indexs[i * n + n_train:end]), axis=0)
            else:
                loop = (i*n+n_train) - end
                train = np.concatenate((indexs[0:loop], indexs[i*n: min(end, i*n+n_train)]), axis=0)
                val = indexs[loop:i * n]
            # train = indexs[i*n: min(end, i*n+n_train)]
            # val = np.concatenate((indexs[0:i*n], indexs[i*n+n_train:end]), axis=0)
            print(len(train), len(val))
            trains.append([volumes[j] for j in train])
            vals.append([volumes[j] for j in val])

        select_data = dict(train=trains,
                           val=vals,
                           test=[volumes[i] for i in indexs[n_train+n_val:]])
        if modality == 't1out' and split.__contains__('t1in'):
            split[modality] = split['t1in']
        elif modality == 't1in' and split.__contains__('t1out'):
            split[modality] = split['t1out']
        else:
            split[modality] = select_data
        print(split[modality])
        print()

        # test whether split rightly
        temp = {x: 0 for x in volumes}
        for i in range(5):
            for k in split[modality]['test']:
                temp[k] += 1
            for k in split[modality]['train'][i]:
                temp[k] += 1
            for k in split[modality]['val'][i]:
                temp[k] += 1
            
            for k, v in temp.items():
                # print("k is : {} v is : {}".format(k,v))
                assert v == 1, k 
            temp = {x: 0 for x in volumes}

    prefix = [str(i) for i in ratios]
    prefix = ''.join(prefix)
    
    write_yaml(split, pjoin(save_root, config.split_yaml))


if __name__ == '__main__':
    
    debug = False

    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # to_png(config.bimod_root, config.png_root, debug)
    split_train_val_test(config.png_root, config.png_root)
