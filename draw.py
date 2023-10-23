import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.backends.cudnn
import numpy as np
import os
import anomaly_dataset
import utils

import yaml

import random

import pickle

import sklearn

import pandas as pd

from tqdm import tqdm


def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

# yaml file load

param = yaml_load()

print('param information')

print(param)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
if device == 'cuda':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

mtype = param['machine_type']
bg = param['bg']
rseed = param['rseed']

input_feature = param['input_feature']

set_randomseed = True
if set_randomseed:
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    # torch.cuda.manual_seed_all(1) # Activate this line when you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(rseed)
    torch.backends.cudnn.enabled = False
    random.seed(rseed)

# dataset generation

# for fast
if param['short_cut'] and os.path.isfile(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/train_dataset.pkl'):
    print("train / valid / test dataset loaded from prepared dataset")

    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/test_dataset.pkl', 'rb') as f3:
        test_dataset = pickle.load(f3)
        f3.close()
else:
    print("error")


for ifile in tqdm(range(len(test_dataset)), desc = "drawing test dataset"):
    #utils.draw_sinIPDIID(test_dataset[ifile][2][0].split('/')[-1][:-4], test_dataset[ifile][2][1].split('/')[-1][:-4], param['machine_type'], param['bg'], test_dataset[ifile][2][0].split('/')[-1][:-4]+'_'+test_dataset[ifile][2][1].split('/')[-1][:-4])

    utils.draw_complex_spectrum(test_dataset[ifile][2][0].split('/')[-1][:-4], test_dataset[ifile][2][1].split('/')[-1][:-4], param['machine_type'], param['bg'], test_dataset[ifile][2][0].split('/')[-1][:-4]+'_'+test_dataset[ifile][2][1].split('/')[-1][:-4])