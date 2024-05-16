# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from copy import deepcopy
from torchvision.transforms import transforms

from config import Modality, num_workers, split_yaml
from misc.utils import read_yaml
from data_loader import externalTransforms as extt


class BaseDataset(Dataset):
    def __init__(self, data_root, phase, modal, fold=0, load_in_ram=False,
                 img_transform=None, msk_transform=None, joint_transform=None):
        super(BaseDataset, self).__init__()
        self.data_root = data_root
        self.phase = phase
        self.fold = fold
        self.load_in_ram = load_in_ram
        self.modal = None
        self.samples = self.__load(phase, modal)
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.joint_transform = joint_transform

    def __load(self, phase, modal):
        samples = []
        self.modal = list(Modality.__members__) if modal == 'all' else [modal]
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
        return samples

    def __len__(self):
        return len(self.samples)

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


def get_loader(data_root, phase, fold, modal, batch_size, data_aug=None, load_in_ram=True):
    joint_augs, img_augs, msk_augs = parse_aug(data_aug)
    if phase == 'train' or phase == 'val':
        dataset = BaseDataset(data_root, phase, fold, modal, load_in_ram=load_in_ram,
                              joint_transform=joint_augs, img_transform=img_augs, msk_transform=msk_augs)
    else:
        dataset = BaseDataset(data_root, phase, fold, modal, load_in_ram=load_in_ram,
                              joint_transform=None, img_transform=img_augs, msk_transform=msk_augs)
    print(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(phase == 'train'),
                        num_workers=num_workers, pin_memory=True, drop_last=(phase == 'train'))
    return loader


def parse_aug(data_aug):
    if data_aug is None or not data_aug:
        img_augs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        msk_augs = extt.MaskToTensor()
        return None, img_augs, msk_augs

    joint_augs = []
    if data_aug['rotate']:
        joint_augs.append(extt.JointRotate(data_aug['rotate_degrees']))
    if data_aug['elasticDeform']:
        joint_augs.append(extt.JointElasticDeform(data_aug['elasticDeform_sigmas'], data_aug['elasticDeform_points']))
    if data_aug['resizeCrop']:
        joint_augs.append(extt.JointRandomResizedCrop(data_aug['resizeCrop_size']))
    joint_augs = extt.JointCompose(joint_augs)

    img_augs = []
    if data_aug['colorJitter']:
        img_augs.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.125))
    if data_aug['gammaCorrect']:
        img_augs.append(extt.RandomGammaCorrection(data_aug['gammaCorrect_gammas']))
    img_augs.append(transforms.ToTensor())
    img_augs.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    img_augs = transforms.Compose(img_augs)

    msk_augs = extt.MaskToTensor()
    return joint_augs, img_augs, msk_augs


if __name__ == '__main__':
    base_dataset = BaseDataset('***/png_dataset',
                               'train', 't1in')
    print(base_dataset)
    exit()
