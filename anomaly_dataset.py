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
import glob
import random

import librosa
from librosa.core.audio import __audioread_load

from tqdm import tqdm

STATUS_LIST = ['normal', 'aL', 'aM', 'aH', 'bL', 'bM', 'bH', 'cL', 'cM', 'cH', 'dL', 'dM', 'dH', 'noise']


def anomaly_dataset(machine_type='toycar', bg = 1, rseed=0, excludes = []):
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
    # this returns numpy array of file list
    # input

    # machine_type = toycar / toytrain
    # bg = 1 / 2 / 3 / 4
    # rseed is randomseed

    # output
    # should return train data, valid data, test data

    # exclude
    exclude_idxs = [STATUS_LIST.index(iexclude) for iexclude in excludes]

    if machine_type == 'toycar':

        # load file list from ./processed_data/{machine_type}
        # here, same index with microphone should be paired


        flist = sorted(glob.glob(f'./processed_data/ToyCar/N{bg}/*'))
        random.shuffle(flist)
        paired_flist = flist2paired_list(flist)

        # separate above np array in to train / valid / test
        # here, train / valid / test should have similar ratio of normal / each anomlay / noise

        filtered_paired_flist = filter_paired_list(paired_flist) # return normal / each anomaly / noise separated pair list. [14 * # * 2] multi dimensional array
        each_length = []
        for ilen in range(len(filtered_paired_flist)):
            each_length.append(len(filtered_paired_flist[ilen]))

        # hyper parameter for dividing train / valid / test
        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 0.2

        # build train / valid / test file list

        train_file_list = []
        valid_file_list = []
        test_file_list = []

        for istatus in range(len(filtered_paired_flist)):

            train_len = int(train_ratio * each_length[istatus])
            valid_len = int(valid_ratio * each_length[istatus])

            if istatus not in exclude_idxs:

                train_file_list += filtered_paired_flist[istatus][:train_len]

                valid_file_list += filtered_paired_flist[istatus][train_len:train_len+valid_len]

            test_file_list += filtered_paired_flist[istatus][train_len+valid_len:]

        # load audio files directly
        # here, each list will be N * 2 * T

        train_audio_list = []
        valid_audio_list = []
        test_audio_list = []

        for ifidx in tqdm(range(len(train_file_list)), desc = "train_audio_loading"):

            single_flist = []

            for imidx in range(2):
                sig, fs = __audioread_load(train_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            train_audio_list.append(single_flist)

        for ifidx in tqdm(range(len(valid_file_list)), desc = "valid_audio_loading"):

            single_flist = []

            for imidx in range(2):
                sig, fs = __audioread_load(valid_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            valid_audio_list.append(single_flist)

        for ifidx in tqdm(range(len(test_file_list)), desc = "test_audio_loading"):

            single_flist = []

            for imidx in range(2):
                sig, fs = __audioread_load(test_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            test_audio_list.append(single_flist)

        # convert to numpy array

        train_audio_list_np = np.asarray(train_audio_list)
        valid_audio_list_np = np.asarray(valid_audio_list)
        test_audio_list_np = np.asarray(test_audio_list)

        return train_audio_list_np, valid_audio_list_np, test_audio_list_np, train_file_list, valid_file_list, test_file_list

    elif machine_type == 'toytrain':


        # load file list from ./processed_data/{machine_type}
        # here, same index with microphone should be paired


        flist = sorted(glob.glob(f'./processed_data/ToyTrain/N{bg}/*'))
        random.shuffle(flist)
        paired_flist = flist2paired_list(flist)

        # separate above np array in to train / valid / test
        # here, train / valid / test should have similar ratio of normal / each anomlay / noise

        filtered_paired_flist = filter_paired_list(paired_flist) # return normal / each anomaly / noise separated pair list. [14 * # * 2] multi dimensional array
        each_length = []
        for ilen in range(len(filtered_paired_flist)):
            each_length.append(len(filtered_paired_flist[ilen]))

        # hyper parameter for dividing train / valid / test
        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 0.2

        # build train / valid / test file list

        train_file_list = []
        valid_file_list = []
        test_file_list = []

        for istatus in range(len(filtered_paired_flist)):

            train_len = int(train_ratio * each_length[istatus])
            valid_len = int(valid_ratio * each_length[istatus])

            if istatus not in exclude_idxs:

                train_file_list += filtered_paired_flist[istatus][:train_len]

                valid_file_list += filtered_paired_flist[istatus][train_len:train_len+valid_len]

            test_file_list += filtered_paired_flist[istatus][train_len+valid_len:]

        # load audio files directly
        # here, each list will be N * 2 * T

        train_audio_list = []
        valid_audio_list = []
        test_audio_list = []

        for ifidx in tqdm(range(len(train_file_list)), desc = "train_audio_loading"):

            single_flist = []

            for imidx in range(4):
                sig, fs = __audioread_load(train_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            train_audio_list.append(single_flist)

        for ifidx in tqdm(range(len(valid_file_list)), desc = "valid_audio_loading"):

            single_flist = []

            for imidx in range(4):
                sig, fs = __audioread_load(valid_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            valid_audio_list.append(single_flist)

        for ifidx in tqdm(range(len(test_file_list)), desc = "test_audio_loading"):

            single_flist = []

            for imidx in range(4):
                sig, fs = __audioread_load(test_file_list[ifidx][imidx], offset=0.0, duration=None, dtype=np.float32)
                single_flist.append(sig)

            test_audio_list.append(single_flist)

        # convert to numpy array

        train_audio_list_np = np.asarray(train_audio_list)
        valid_audio_list_np = np.asarray(valid_audio_list)
        test_audio_list_np = np.asarray(test_audio_list)

        return train_audio_list_np, valid_audio_list_np, test_audio_list_np, train_file_list, valid_file_list, test_file_list


    else:
        print("machine type should be either toycar or toytrain")
        return -1 # warning


def flist2paired_list(file_list):

    # modify list of files to list of pairs of files
    only_idx_info = [afile2only_idx(afile) for afile in file_list]
    only_idx_info_set = list(dict.fromkeys(only_idx_info)) # preserve order

    # pairing

    paired_list = []

    for iitem in range(len(only_idx_info_set)):
        pair_item = only_idx_info_set[iitem]
        apair = [file_list[iiitem] for iiitem, xiitem in enumerate(only_idx_info) if xiitem == pair_item]

        paired_list.append(apair)

    return paired_list

def afile2only_idx(afile):

    return '_'.join([afile.split('/')[-1].split('_')[0], afile.split('/')[-1].split('_')[2], afile.split('/')[-1].split('_')[4]])


def filter_paired_list(paired_flist):

    status_list = [pair2status_info(apair) for apair in paired_flist]

    filter_list = []

    # built filter_list using information from status_list
    for istatus in STATUS_LIST:

        afilter_list = []

        for idx in range(len(status_list)):

            if status_list[idx] == istatus:
                afilter_list.append(paired_flist[idx])

        filter_list.append(afilter_list)


    return filter_list

def pair2status_info(apair):

    first_one = apair[0]
    last_folder = first_one.split('/')[-1]

    is_noise_only_period = ('noise_only_period' in last_folder)

    if last_folder[1] == 'N' and not is_noise_only_period: #in case of normal
        status_info = 'normal'
    elif last_folder[1] != 'N' and not is_noise_only_period: # in case of anomaly
        status_info = last_folder.split('_')[0].split('-')[-2] + last_folder.split('_')[0].split('-')[-1][-1]
    else:
        status_info = 'noise'

    return status_info