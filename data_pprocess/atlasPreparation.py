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

from misc.utils import maybe_mkdir
from misc.visualize import colorize


class AtlasPreparation(object):
    def __init__(self, root):
        self.root = root
        self.label_range = (0,  # background
                            6,  # liver
                            2,  # right kidney
                            3,  # left kidney
                            1)  # spleen

    def __collect_data_dict(self, new_spacing, crop_size):
        data_dict = {}
        img_root = pjoin(self.root, 'Training', 'img')
        lbl_root = pjoin(self.root, 'Training', 'label')

        for pid in tqdm(os.listdir(pjoin(self.root, 'Training', 'img'))):
            # Covert to nii.gz.
            # 1. Read image and label.
            image = sitk.ReadImage(pjoin(img_root, pid))
            label = sitk.ReadImage(pjoin(lbl_root, pid.replace('img', 'label')))

            label = sitk.GetArrayFromImage(label)
            mask = np.zeros(label.shape, dtype=np.uint8)
            for i, origin_label in enumerate(self.label_range):
                if i == 0:
                    continue
                mask[label == origin_label] = i

            # 2. Remove the unrelated region in z-axis which have no given label.
            has_label = np.sum(mask != 0, axis=(1, 2)) != 0
            start, end = len(has_label) - 1, 0
            for i, hl in enumerate(has_label):
                start = min(i, start) if hl else start
                end = max(i, end) if hl else end
            if start >= end:
                continue
            mask = mask[start: end+1, :, :]
            image = image[:, :, start: end+1]

            label = sitk.GetImageFromArray(mask)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

            image = image[:, ::-1, :]
            label = label[:, ::-1, :]

            # 3. Re-sample to the same spacing.
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

            # 4. Center crop.
            dx = (new_size[0] - crop_size) // 2
            dy = (new_size[1] - crop_size) // 2
            image = image[dx: dx+crop_size, dy: dy+crop_size, :]
            label = label[dx: dx+crop_size, dy: dy+crop_size, :]

            # 5. Put out to a dict.
            data_dict[f'ct_{pid[4:7]}'] = (image, label)
        return data_dict

    def run(self, save_root, new_spacing, crop_size):
        modality_root = pjoin(save_root, 'ct')
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
    chaos_pr = AtlasPreparation(config.atlas_root)
    chaos_pr.run(config.bimod_root, new_spacing=config.new_spacing, crop_size=config.input_size)

