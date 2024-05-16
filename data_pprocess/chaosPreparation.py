# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

from os.path import join as pjoin
import SimpleITK as sitk
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import warnings

from misc.utils import maybe_mkdir
from misc.visualize import colorize


class ChaosPprocess(object):
    def __init__(self, root, modality):
        assert modality in ('t1in', 't1out', 't2')
        self.root = root
        self.modality = modality
        self.label_range = ((0, 0),      # background
                            (55, 70),    # liver
                            (110, 135),  # right kidney
                            (175, 200),  # left kidney
                            (240, 255))  # spleen

    def __collect_data_dict(self, new_spacing, crop_size):
        data_dict = {}

        for pid in tqdm(os.listdir(pjoin(self.root, 'MR'))):
            if self.modality == 't1in':
                img_root = pjoin(self.root, 'MR', pid, 'T1DUAL', 'DICOM_anon', 'InPhase')
                lbl_root = pjoin(self.root, 'MR', pid, 'T1DUAL', 'Ground')
            elif self.modality == 't1out':
                img_root = pjoin(self.root, 'MR', pid, 'T1DUAL', 'DICOM_anon', 'OutPhase')
                lbl_root = pjoin(self.root, 'MR', pid, 'T1DUAL', 'Ground')
            elif self.modality == 't2':
                img_root = pjoin(self.root, 'MR', pid, 'T2SPIR', 'DICOM_anon')
                lbl_root = pjoin(self.root, 'MR', pid, 'T2SPIR', 'Ground')
            else:
                raise ValueError

            # Covert a list of .dcm/.png files to nii.gz.
            # 1. Read image and label.
            reader = sitk.ImageSeriesReader()
            dcm_names = reader.GetGDCMSeriesFileNames(img_root)
            reader.SetFileNames(dcm_names)
            image = reader.Execute()

            lbls = []
            for png in sorted(os.listdir(lbl_root)):
                lbl = imread(pjoin(lbl_root, png))
                lbls.append(lbl)
            label = np.array(lbls)
            mask = np.zeros(label.shape, dtype=np.int8)
            for i, (mi, ma) in enumerate(self.label_range):
                if i == 0:
                    continue
                mask[(label >= mi) & (label <= ma)] = i
                # print(((label >= mi) & (label <= ma)).sum())

            label = sitk.GetImageFromArray(mask)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

            # 2. Re-sample to the same spacing.
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())

            old_spacing = image.GetSpacing()
            old_size = image.GetSize()

            new_size = [osz * osp / nsp for osz, nsp, osp in zip(old_size, new_spacing, old_spacing)]
            new_size = list(map(int, new_size))
            for i in range(2):
                new_size[i] = max(crop_size, new_size[i])  # if smaller than given size
            round_new_spacing = [osp * osz / nsz for osp, nsz, osz in zip(old_spacing, new_size, old_size)]
            # print(old_size, new_size, old_spacing, new_spacing)

            resampler.SetOutputSpacing(round_new_spacing)
            resampler.SetSize(new_size)

            # resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetInterpolator(sitk.sitkBSpline)
            image = resampler.Execute(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            label = resampler.Execute(label)
            # print(image.GetSize(), label.GetSize(), image.GetSpacing(), label.GetSpacing())

            # 3. Center crop.
            dx = (new_size[0] - crop_size) // 2
            dy = (new_size[1] - crop_size) // 2
            image = image[dx: dx+crop_size, dy: dy+crop_size, :]
            label = label[dx: dx+crop_size, dy: dy+crop_size, :]

            # 4. Put out to a dict.
            data_dict[f'{self.modality}_{pid.rjust(3, "0")}'] = (image, label)
        return data_dict

    def run(self, save_root, new_spacing, crop_size):
        modality_root = pjoin(save_root, self.modality)
        img_root = pjoin(modality_root, 'images')
        lbl_root = pjoin(modality_root, 'labels')

        maybe_mkdir(save_root, modality_root, img_root, lbl_root)
        data_dict = self.__collect_data_dict(new_spacing, crop_size)
        for pid, (image, label) in data_dict.items():
            save_img_path = pjoin(img_root, f'{pid}.nii.gz')
            save_lbl_path = pjoin(lbl_root, f'{pid}.nii.gz')
            sitk.WriteImage(image, save_img_path)
            sitk.WriteImage(label, save_lbl_path)
        return


if __name__ == '__main__':
    import config
    maybe_mkdir(config.base_root)
    for modality in ('t1in', 't1out', 't2'):
        # if modality == 't2':
        #     warnings.warn('T2 case31 has some label error!')
        chaos_pr = ChaosPprocess(config.chaos_root, modality)
        chaos_pr.run(config.bimod_root, new_spacing=config.new_spacing, crop_size=config.input_size)
