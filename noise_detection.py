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
import noise_detection_dataset

import torch.nn.functional as F
import yaml

import random
import utils
import pickle

import sklearn

import pandas as pd

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
    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/train_dataset.pkl', 'rb') as f1:
        train_dataset = pickle.load(f1)
        f1.close()

    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/valid_dataset.pkl', 'rb') as f2:
        valid_dataset = pickle.load(f2)
        f2.close()

    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/test_dataset.pkl', 'rb') as f3:
        test_dataset = pickle.load(f3)
        f3.close()
else:
    train_data, valid_data, test_data, train_audio_name, valid_audio_name, test_audio_name = noise_detection_dataset.noise_detection_dataset(machine_type=param['machine_type'], bg = param['bg'], rseed=param['rseed'], excludes = param['exclude'])

    train_dataset = []

    for iaudio in range(len(train_audio_name)):
        if noise_detection_dataset.pair2status_info(train_audio_name[iaudio]) == 'noise':
            a_label = np.ones(2)/2 * 1.0
        else:
            a_label = np.zeros(2)
            a_label[0] = 1.0

        # shuffle

        # random shuffle
        random_permute = np.random.permutation(2)
        a_data = train_data[iaudio,:,:]
        a_data = a_data[random_permute]
        a_label = a_label[random_permute]


        # append
        train_dataset.append([a_data, a_label])

    # this is for valid dataset

    valid_dataset = []

    for iaudio in range(len(valid_audio_name)):
        if noise_detection_dataset.pair2status_info(valid_audio_name[iaudio]) == 'noise':
            a_label = np.ones(2) / 2 * 1.0
        else:
            a_label = np.zeros(2)
            a_label[0] = 1.0

        # shuffle

        # random shuffle
        random_permute = np.random.permutation(2)
        a_data = valid_data[iaudio, :, :]
        a_data = a_data[random_permute]
        a_label = a_label[random_permute]

        # append
        valid_dataset.append([a_data, a_label])

    # this is for train dataset

    test_dataset = []

    for iaudio in range(len(test_audio_name)):
        if noise_detection_dataset.pair2status_info(test_audio_name[iaudio]) == 'noise':
            a_label = np.ones(2) / 2 * 1.0
        else:
            a_label = np.zeros(2)
            a_label[0] = 1.0

        # shuffle

        # random shuffle
        random_permute = np.random.permutation(2)
        a_data = test_data[iaudio, :, :]
        a_data = a_data[random_permute]
        a_label = a_label[random_permute]

        # append
        test_dataset.append([a_data, a_label, test_audio_name[iaudio]])


    os.makedirs(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}', exist_ok=True)
    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/train_dataset.pkl', 'wb') as f1:
        pickle.dump(train_dataset, f1)
        f1.close()

    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/valid_dataset.pkl', 'wb') as f2:
        pickle.dump(valid_dataset, f2)
        f2.close()

    with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/test_dataset.pkl', 'wb') as f3:
        pickle.dump(test_dataset, f3)
        f3.close()


for idata in range(len(train_dataset)):

    # modifying label, allocating new label
    if max(train_dataset[idata][1]) == 1: # machine_sound
        a_label = np.zeros(2)
        a_label[0] = 1
        train_dataset[idata][1] = a_label
    elif max(train_dataset[idata][1]) == 0.5: # noise
        a_label = np.zeros(2)
        a_label[1] = 1
        train_dataset[idata][1] = a_label
    else:
        print("something wrong.")

# for valid data

for idata in range(len(valid_dataset)):

    # modifying label, allocating new label
    if max(valid_dataset[idata][1]) == 1: # machine_sound
        a_label = np.zeros(2)
        a_label[0] = 1
        valid_dataset[idata][1] = a_label
    elif max(valid_dataset[idata][1]) == 0.5: # noise
        a_label = np.zeros(2)
        a_label[1] = 1
        valid_dataset[idata][1] = a_label
    else:
        print("something wrong.")

# for test data

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

# transfer dataset to dataloader

train_dataloader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model

if input_feature == 'intensity_and_phase_spectrum':

    detector = utils.ResNet_CLF_4ch(2, device).to(device)

elif input_feature == 'sinIPD':

    detector = utils.ResNet_sinIPD_long(2, device).to(device)

elif input_feature == 'IID+sinIPD':

    detector = utils.ResNet_sinIPDIID_long(2, device).to(device)

elif input_feature == 'IPD':

    detector = utils.ResNet_IPD_long(2, device).to(device)

elif input_feature == 'IID':

    detector = utils.ResNet_IID_long(2, device).to(device)

# loss function

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(detector.parameters(), lr=1e-5)

best_model = detector
best_val_f1 = 0
best_val_thres = 0

val_f1s = []

for t in range(param['epochs']):
    detector.train()
    print(f"Epoch {t + 1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        #y = y.float().to(device)
        y = y.to(device)
        # lbsm


        pred = detector(X)
        pred = F.softmax(pred, dim=1) # deactivate if not IPD
        loss = -torch.mean(torch.log(pred)*y) # CCE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    detector.eval()

    valid_labels = []
    valid_predone = [] # about classification label predicted in sv manner
    for batch, (X, y) in enumerate(valid_dataloader):

        pred = detector(X).cpu().detach()
        pred = F.softmax(pred, dim=1)  # deactivate if not IPD

        if float(pred[0][0]) >= 0.5: # in case of machine sound
            valid_predone.append(0)
        else:
            valid_predone.append(1)


        if float(y[0][1]) == 1: #this is noise
            valid_labels.append(1)

        else:
            valid_labels.append(0)


    valid_max_f1 = sklearn.metrics.f1_score(valid_labels, valid_predone)


    print(f'valid f1: {valid_max_f1}')

    val_f1s.append(valid_max_f1)
    if valid_max_f1 >= best_val_f1: # threshold process should be added
        best_model = detector
        best_val_f1 = valid_max_f1

    print(f'best valid f1: {best_val_f1}')

# test

test_labels = []
test_names = []
test_pred_labels = []
for batch, (X, y, name) in enumerate(test_dataloader):

    pred = best_model(X).cpu().detach()

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

os.makedirs(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}', exist_ok=True)

with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}/valid_f1s.txt','w') as f_val:
    f_val.write(str(val_f1s))
    f_val.close()

with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}/test_f1s.txt','w') as f_test:
    f_test.write(str(test_f1))
    f_test.close()

with open(f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}/params.txt','w') as f_param:
    print(param, file=f_param)
    f_param.close()

torch.save(best_model, f'./dataset_prepared/{mtype}/N{bg}/{rseed}/result/{input_feature}/best_model')
