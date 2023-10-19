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


import yaml

import random

import pickle

import sklearn

import pandas as pd

def yaml_load():
    with open("infer2others.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

# yaml file load

param = yaml_load()

print('param information')

print(param)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

mtype = param['machine_type']
bg_from = param['bg_from']
bg_to = param['bg_to']

rseed = param['rseed']

model_type = param['model_type']

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
if os.path.isfile(f'./dataset_prepared/{mtype}/N{bg_to}/{rseed}/train_dataset.pkl'):
    print("prepare test dataset")

    with open(f'./dataset_prepared/{mtype}/N{bg_to}/{rseed}/test_dataset.pkl', 'rb') as fdata:
        test_dataset = pickle.load(fdata)
        fdata.close()


if os.path.isfile(f'./dataset_prepared/{mtype}/N{bg_from}/{rseed}/result/{model_type}/best_model'):
    print("load trained model")

    test_model = torch.load(f'./dataset_prepared/{mtype}/N{bg_from}/{rseed}/result/{model_type}/best_model')

test_model.eval()



for idata in range(len(test_dataset)):

    # modifying label, allocating new label
    if max(test_dataset[idata][1]) == 1: # machine_sound
        a_label = np.zeros(2)
        a_label[0] = 1
        test_dataset[idata][1] = a_label
    elif max(test_dataset[idata][1]) == 0.5: # noise
        a_label = np.zeros(2)
        a_label[1] = 1
        test_dataset[idata][1] = a_label
    else:
        print("something wrong.")

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)



test_labels = []
test_names = []
test_pred_labels = []
for batch, (X, y, name) in enumerate(test_dataloader):

    pred = test_model(X).cpu().detach()

    pred_label = (pred[0][0] < 0.5)


    test_pred_labels.append(pred_label.item())

    if float(y[0][1]) == 1: # this is noise
        test_labels.append(1)

    else:
        test_labels.append(0)

    test_names.append(name)



test_f1 = sklearn.metrics.f1_score(test_labels, test_pred_labels)
print(f'test f1: {test_f1}')



# saving results

os.makedirs(f'./infer2others/{mtype}/from_N{bg_from}_to_N_{bg_to}/{rseed}/result/{model_type}', exist_ok=True)

with open(f'./infer2others/{mtype}/from_N{bg_from}_to_N_{bg_to}/{rseed}/result/{model_type}/test_f1s.txt','w') as f_test:
    f_test.write(str(test_f1))
    f_test.close()

with open(f'./infer2others/{mtype}/from_N{bg_from}_to_N_{bg_to}/{rseed}/result/{model_type}/params.txt','w') as f_param:
    print(param, file=f_param)
    f_param.close()
