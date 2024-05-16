# -*- coding: utf-8 -*-

import sys 
import os
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List
import random
from PIL import Image
from copy import deepcopy

from config import Modality, num_workers, split_yaml
from misc.utils import read_yaml
from data_loader.baseLoader import parse_aug


class BalanceDataset(Dataset):
    def __init__(self, data_root, phase, fold=0, load_in_ram=False,
                 img_transform=None, msk_transform=None, joint_transform=None):
        super(BalanceDataset, self).__init__()
        self.data_root = data_root
        self.phase = phase
        self.modal = None
        self.fold = fold
        self.load_in_ram = load_in_ram
        self.samples, self.modal_sample_ids, self.n = self.__load(phase)
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.joint_transform = joint_transform

    def __load(self, phase):
        samples, n = [], 0
        self.modal = list(Modality.__members__)
        modal_sample_ids = [[] for _ in self.modal]
        split = read_yaml(pjoin(self.data_root, split_yaml))
        for m in self.modal:
            modal_root = pjoin(self.data_root, m)

            temp = split[m][phase] if phase == 'test' else split[m][phase][self.fold]
            
            for pid in temp:
                pid_root = pjoin(modal_root, pid, 'images')
                for png in sorted(os.listdir(pid_root)):
                    img = pjoin(pid_root, png)  # eg. /path/to/ct/001/ct_001_000.png
                    msk = img.replace('images', 'labels')
                    if self.load_in_ram:
                        img = deepcopy(Image.open(img))
                        msk = deepcopy(Image.open(msk))
                    samples.append((img, msk, Modality[m].value, png.replace('.png', '')))
                    modal_sample_ids[Modality[m].value].append(n)
                    n += 1

        assert n == len(samples)
        return samples, modal_sample_ids, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, msk, mdl, inm = self.samples[i]
        if not self.load_in_ram:
            img = Image.open(img); msk = Image.open(msk)
        if self.joint_transform:
            img, msk = self.joint_transform(img, msk)
        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)
        return img, msk, mdl, inm

    def __repr__(self):
        format_string = self.__class__.__name__ + '(samples={0}, phase={1} {2}, modality={3})'.format(
            len(self.samples), self.phase, self.fold, self.modal)
        # format_string += str(self.joint_transform) if self.joint_transform else 'None'
        # format_string += str(self.img_transform) + '\n' if self.img_transform else 'None,\n'
        # format_string += str(self.msk_transform) if self.msk_transform else 'None'
        return format_string


class ModalityBalanceBatchSampler(Sampler):
    def __init__(self, samples: List[List[int]], batch_size: int):
        self.samples = samples
        self.num_modality = len(samples)
        self.batch_size = batch_size
        self.num_samples_per_modality = self.batch_size // self.num_modality
        self.starts = [0 for _ in range(self.num_modality)]
        self.n = 0

        for i, spl in enumerate(self.samples):
            self.n = max(self.n, len(spl))
            random.shuffle(self.samples[i])

    def __iter__(self):
        for _ in range(0, self.n, self.num_samples_per_modality):
            batch = []
            for j, spl in enumerate(self.samples):
                s = self.starts[j]
                batch.extend(spl[s: s+self.num_samples_per_modality])

                self.starts[j] += self.num_samples_per_modality
                if self.starts[j] > len(spl):
                    random.shuffle(self.samples[j])
                    self.starts[j] = 0

            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        return self.n // self.num_samples_per_modality


def get_loader(data_root, phase, fold, batch_size, data_aug=None, load_in_ram: bool = True):
    joint_augs, img_augs, msk_augs = parse_aug(data_aug)
    if phase == 'train' or phase == 'val':
        dataset = BalanceDataset(data_root, phase, fold, load_in_ram=load_in_ram,
                                 joint_transform=joint_augs, img_transform=img_augs, msk_transform=msk_augs)
    else:
        raise ValueError
    print(dataset)

    num_modality = len(Modality.__members__)
    assert batch_size % num_modality == 0, 'Batch size must be an integral multiple of #modality.'
    batch_sampler = ModalityBalanceBatchSampler(dataset.modal_sample_ids, batch_size)
    loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, pin_memory=True)
    return loader


if __name__ == '__main__':
    from config import png_root, data_aug
    loader = get_loader(png_root, 'train', 8, data_aug)

    # for i, (img, msk, mdl, inm) in enumerate(loader):
    #     print(i, mdl)
    # for i, (img, msk, mdl, inm) in enumerate(loader):
    #     print(i, mdl)

    itr = iter(loader)
    for i in range(500):
        try:
            img, msk, mdl, inm = next(itr)
        except StopIteration:
            print()
            itr = iter(loader)
            img, msk, mdl, inm = next(itr)
        print(i, inm)

