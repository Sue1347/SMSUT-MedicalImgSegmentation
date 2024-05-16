# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader, Sampler
from typing import List
import random

import config as cfg
from data_loader.baseLoader import parse_aug
from data_loader.balanceLoader import BalanceDataset


class InTurnTrainBatchSampler(Sampler):
    def __init__(self, samples: List[List[int]], batch_size: int, shuffle: bool):
        """ shuffle: whether shuffles the order of data stream eg 0 1 2 3 to random combination, eg 3 1 2 0.
        """
        self.samples = samples
        self.num_modality = len(samples)
        self.batch_size = batch_size
        self.starts = [0 for _ in range(self.num_modality)]
        
        self.shuffle = shuffle
        self.queue = [i for i in range(self.num_modality)]
        self.cur_modality = 0

        self.n = 0
        max_batch_per_modality = 0
        for i, spl in enumerate(self.samples):
            n = len(spl) // self.batch_size - 1 if len(spl) % self.batch_size else len(spl) // self.batch_size
            max_batch_per_modality = max(n, max_batch_per_modality)
            random.shuffle(self.samples[i])
        self.n = self.num_modality * max_batch_per_modality
        
    def __iter__(self):
        for _ in range(self.n):
            if not self.shuffle:
                cur_modality = self.cur_modality
            else:
                cur_modality = self.queue[self.cur_modality]

            s = self.starts[cur_modality]
            if s + self.batch_size >= len(self.samples[cur_modality]):
                self.starts[cur_modality] = 0
                s = 0
                random.shuffle(self.samples[cur_modality])
            else:
                self.starts[cur_modality] += self.batch_size

            batch = self.samples[cur_modality][s: s+self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

            if self.shuffle and self.cur_modality + 1 == self.num_modality:
                random.shuffle(self.queue)
            self.cur_modality = (self.cur_modality + 1) % self.num_modality
            
    def __len__(self):
        return self.n


class InTurnTestBatchSampler(Sampler):
    def __init__(self, samples: List[List[int]], batch_size: int):
        self.samples = samples
        self.num_modality = len(samples)
        self.batch_size = batch_size

        self.n = 0
        for _, spl in enumerate(self.samples):
            self.n += len(spl) // self.batch_size

    def __iter__(self):
        for spl in self.samples:
            for i in range(0, len(spl), self.batch_size):
                batch = spl[i: i+self.batch_size]
                yield batch

    def __len__(self):
        return self.n


def get_loader(data_root, phase, fold, batch_size, data_aug=None, load_in_ram: bool = True):
    joint_augs, img_augs, msk_augs = parse_aug(data_aug)
    if phase == 'train' or phase == 'val':
        dataset = BalanceDataset(data_root, phase, fold, load_in_ram=load_in_ram,
                                 joint_transform=joint_augs, img_transform=img_augs, msk_transform=msk_augs)
        batch_sampler = InTurnTrainBatchSampler(dataset.modal_sample_ids, batch_size, shuffle=False)
        loader = DataLoader(dataset, num_workers=cfg.num_workers, batch_sampler=batch_sampler, pin_memory=True)
    else:
        dataset = BalanceDataset(data_root, phase, fold, load_in_ram=load_in_ram,
                                 joint_transform=None, img_transform=img_augs, msk_transform=msk_augs)
        batch_sampler = InTurnTestBatchSampler(dataset.modal_sample_ids, batch_size)
        loader = DataLoader(dataset, num_workers=cfg.num_workers, batch_sampler=batch_sampler, pin_memory=True)
    print(dataset)

    return loader


if __name__ == '__main__':
    from config import data_aug, png_root
    loader = get_loader(png_root, 'train', 1, 4, None)

    
    for i in range(2):
        itr = iter(loader)
        for j in range(cfg.num_iter_per_epoch):
            try:
                _, _, mdl, _ = next(itr)
            except StopIteration:
                itr = iter(loader)
                _, _, mdl, _ = next(itr)

            print(i, j, mdl)
        del itr
    
    # for i, (img, msk, mdl, inm) in enumerate(loader):
    #     print(i, mdl, type(mdl), inm)

    # for i, (img, msk, mdl, inm) in enumerate(loader):
    #     print(i, inm)
