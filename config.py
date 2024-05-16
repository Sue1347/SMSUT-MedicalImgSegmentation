# -*- coding: utf-8 -*-

from enum import Enum
import torch

# Different dataset changes different modalities
class Modality(Enum):
    ct = 0
    t1in = 1
    t1out = 2
    t2 = 3

# class Modality(Enum):
#     a = 0
#     b = 1
#     c = 2
#     d = 3
#     e = 4
#     f = 5


# Misc.
seed = 2020
n_modal = len(Modality.__members__)
# TODO: change modality
n_label = 4  # 4---chaos,1---saml

# Training.
num_iter_per_epoch = 150  
max_epoch = 200 
exp_alpha = 1.
weight_dc = 0.5
weight_ce = 0.5

# Network.
img_channels = 1
base_width = 16 

# Pre-process.
atlas_root = '***/Multi-altlas 2015/RawData' # dataset for Multi-altlas 2015/RawData
chaos_root = '***/Chaos 2019/CHAOS_Train_Sets/Train_Sets' # dataset for Chaos 2019/CHAOS_Train_Sets/Train_Sets
saml_root = '***/SAML/' # dataset for SAML

bimod_root = '***/bimod' # the base directory for processed dataset
base_root = '***/bimod'  # the base directory for processed dataset
expr_root = '***/bimod-out' # the base directory for output data
png_root = base_root

new_spacing = (1.5, 1.5, 5)
input_size = 256
mod_type = ('ct, t1in, t1out, t2') 

# Data loader. 
split_yaml = 'semi-1910.yaml' 

batch_size = 8 
# batchsize is 2 for AHDC

num_workers = 6
data_aug = dict(
    rotate=True,
    rotate_degrees=15,
    resizeCrop=True,
    resizeCrop_size=input_size,
    elasticDeform=True,
    elasticDeform_sigmas=(9., 13.),
    elasticDeform_points=3,
    colorJitter=False,
    gammaCorrect=False,
    gammaCorrect_gammas=(0.7, 1.5),
)

# Optimize.
lr = 1e-2  
weight_decay = 1e-3 

# nce loss
nce_layers = [5]  # layers after encoder, before decoder

# coranet
thres = 0.5
default_w = torch.FloatTensor([1, 1])
# saml: torch.FloatTensor([1, 1])
# chaos: torch.FloatTensor([1, 1, 1, 1, 1])
w_con = torch.FloatTensor([1, 5])
# saml: torch.FloatTensor([1, 5])
# chaos: torch.FloatTensor([1, 5, 5, 5, 5])
w_rad = torch.FloatTensor([5, 1])
# saml: torch.FloatTensor([5, 1])
# chaos: torch.FloatTensor([5, 1, 1, 1, 1])

pre_epoch = 100  
cora_epoch = 200 
pred_step = 10

