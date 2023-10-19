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

from librosa.core.audio import __audioread_load

from matplotlib import pyplot as plt


def sigs2ips(signal, device):

    sample_rate = 16000

    win_len_sec = 0.256 #1.024 #0.256
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5 #0.1 #0.5
    nfft = win_len
    fre_used_ratio = 2000 / (sample_rate / 2)

    nsample = signal.shape[2]
    nch = signal.shape[1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    nb = signal.shape[0]
    nt = int((nsample) / win_shift) + 1 # for iSTFT
    # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
    stft = torch.zeros((nb, nf, nt, nch*2), dtype=torch.complex64)

    window = torch.hann_window(window_length=win_len, device=signal.device)
    for ch_idx in range(0, nch, 1):
        stft_temp = torch.stft(signal[:, ch_idx, :], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                               window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

        stft[:, :, :, ch_idx] = 20*torch.log10(torch.abs(stft_temp[:, 0:nf, 0:nt]))
        stft[:, :, :, ch_idx+2] = torch.angle(stft_temp[:, 0:nf, 0:nt])


    stft = torch.real(stft)
    return stft.to(device)


def draw_ips(signal1name, signal2name, machine_type, bg, title, savefig=True, fileformat = '.jpg', abs_path = './processed_data'):

    # draw single signal pairs in to image file (deafult .jpg file)
    # modify code after feed back from prof.


    sample_rate = 16000

    if machine_type == 'toycar':
        machine_type = 'ToyCar'

    elif machine_type == 'toytrain':
        machine_type = 'ToyTrain'

    signal1name = f'{abs_path}/{machine_type}/N{bg}/{signal1name}.wav'
    signal2name = f'{abs_path}/{machine_type}/N{bg}/{signal2name}.wav'

    y_machine1, _ = __audioread_load(signal1name, offset=0.0, duration=None, dtype=np.float32)
    y_machine2, _ = __audioread_load(signal2name, offset=0.0, duration=None, dtype=np.float32)

    y_machines = np.concatenate((y_machine1.reshape(1, 1, -1), y_machine2.reshape(1, 1, -1)), axis=1)

    y_machines = torch.tensor(y_machines)

    IID_drawed = sigs2ips(y_machines, 'cpu')

    plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 15})
    plt.subplot(2,2,1)

    plt.title(f'Intensity_{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])}'+'_'+signal1name.split('/')[-1].split('_')[1])
    plt.imshow(IID_drawed[0,:,:,0], aspect='auto')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')

    freq=IID_drawed.shape[1]
    tm = IID_drawed.shape[2]


    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.colorbar()
    plt.clim(-100,0)
    plt.gca().invert_yaxis()

    plt.subplot(2,2,2)
    plt.title(f'Intensity_{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])}'+'_'+signal2name.split('/')[-1].split('_')[1])
    plt.imshow(IID_drawed[0,:,:,1], aspect='auto')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.colorbar()
    plt.clim(-100, 0)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.subplot(2,2,3)
    plt.title(f'Phase_{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])}'+'_'+signal1name.split('/')[-1].split('_')[1])
    plt.imshow(IID_drawed[0,:,:,2], aspect='auto')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.colorbar()
    plt.clim(-4, 4)
    plt.gca().invert_yaxis()
    plt.tight_layout()


    plt.subplot(2,2,4)
    plt.title(f'Phase_{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])}'+'_'+signal2name.split('/')[-1].split('_')[1])
    plt.imshow(IID_drawed[0,:,:,3], aspect='auto')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.colorbar()
    plt.clim(-4, 4)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(f'./intensity_and_phase_spectrum/{machine_type}/N{bg}/', exist_ok=True)

    plt.savefig(f'./intensity_and_phase_spectrum/{machine_type}/N{bg}/' + title + fileformat) #rseed 0 only plz
    plt.close()
    #print('draw complete!')
    return 0


def drawsinIPDIID(signal1name, signal2name, machine_type, bg, title, savefig=True, fileformat = '.jpg', abs_path = './processed_data'):

    # draw single signal pairs in to image file (deafult .jpg file)
    # modify code after feed back from prof.


    sample_rate = 16000

    if machine_type == 'toycar':
        machine_type = 'ToyCar'

    elif machine_type == 'toytrain':
        machine_type = 'ToyTrain'

    signal1name = f'{abs_path}/{machine_type}/N{bg}/{signal1name}.wav'
    signal2name = f'{abs_path}/{machine_type}/N{bg}/{signal2name}.wav'

    y_machine1, _ = __audioread_load(signal1name, offset=0.0, duration=None, dtype=np.float32)
    y_machine2, _ = __audioread_load(signal2name, offset=0.0, duration=None, dtype=np.float32)

    y_machines = np.concatenate((y_machine1.reshape(1, 1, -1), y_machine2.reshape(1, 1, -1)), axis=1)

    y_machines = torch.tensor(y_machines)

    IID_drawed = sigs2sinIPDIID(y_machines, 'cpu')

    #plt.title(f'{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])}'+'_'+signal1name.split('/')[-1].split('_')[1])
    plt.rcParams.update({'font.size': 15})
    plt.subplot(211)
    plt.title(f'{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])} (sinIPD)')
    plt.imshow(IID_drawed[0, 0, :,:], aspect='auto')
    freq = IID_drawed.shape[2]
    tm = IID_drawed.shape[3]
    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.clim(-1,1)
    plt.gca().invert_yaxis()

    plt.subplot(212)
    plt.title(f'{machine_type}_{bg}_{anomaly_dataset.pair2status_info([signal1name])} (IID)')
    plt.imshow(IID_drawed[0, 1, :,:], aspect='auto')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    plt.yticks(np.arange(0,freq, step=freq-1),labels=['0','2000'])
    plt.xticks(np.arange(0,tm, step=tm-1),labels=['0','12'])
    plt.colorbar()
    plt.clim(-1,1)
    plt.gca().invert_yaxis()

    plt.tight_layout()

    os.makedirs(f'./IID+sinIPD/{machine_type}/N{bg}/', exist_ok=True)

    plt.savefig(f'./IID+sinIPD/{machine_type}/N{bg}/' + title + fileformat) #rseed 0 only plz
    plt.close()
    #print('draw complete!')
    return 0


class ResNet_ips(nn.Module):
    def __init__(self, n_class, device):
        super(ResNet_ips, self).__init__()
        self.n_channel = 8
        self.n_class = n_class
        self.conv1 = nn.Conv2d(4, self.n_channel, kernel_size=7, stride=2, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, x):
        x = sigs2ips(x, self.device)
        x = self.conv1(torch.permute(x,(0,-1,1,2)))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, projection=False):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = projection
        if self.projection:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2, padding=(1, 1))
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        if self.projection:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, stride=2, kernel_size=1)
        else:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            skip = self.downsample(x)
        else:
            skip = x
        out += skip
        out = self.relu(out)
        return out


class ResNet_sinIPD(nn.Module):
    def __init__(self, n_class, device):
        super(ResNet_sinIPD, self).__init__()
        self.n_channel = 8
        self.n_class = n_class
        self.conv1 = nn.Conv2d(1, self.n_channel, kernel_size=7, stride=1, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, x):
        x = sigs2sinIPD(x, self.device)
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x

class ResNet_sinIPDIID(nn.Module):
    def __init__(self, n_class, device):
        super(ResNet_sinIPDIID, self).__init__()
        self.n_channel = 8
        self.n_class = n_class
        self.conv1 = nn.Conv2d(2, self.n_channel, kernel_size=7, stride=1, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, x):
        x = sigs2sinIPDIID(x, self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x

class ResNet_IPD(nn.Module):
    def __init__(self, n_class, device):
        super(ResNet_IPD, self).__init__()
        self.n_channel = 8
        self.n_class = n_class
        self.conv1 = nn.Conv2d(1, self.n_channel, kernel_size=7, stride=1, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, x):
        x = sigs2IPD(x, self.device)
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x


class ResNet_IID(nn.Module):
    def __init__(self, n_class, device):
        super(ResNet_IID, self).__init__()
        self.n_channel = 8
        self.n_class = n_class
        self.conv1 = nn.Conv2d(1, self.n_channel, kernel_size=7, stride=1, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)
        #self.softmax = nn.Softmax(dim=1)
        self.device = device
    def forward(self, x):
        x = sigs2IID(x, self.device)
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, projection=False):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = projection
        if self.projection:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2, padding=(1, 1))
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        if self.projection:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, stride=2, kernel_size=1)
        else:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            skip = self.downsample(x)
        else:
            skip = x
        out += skip
        out = self.relu(out)
        return out


def sigs2IID(signal, device): # can consume cpu memory due to numpy usage, should be changed in long term

    sample_rate = 16000

    win_len_sec = 0.256 #1.024 #0.256
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5 #0.1 #0.5
    nfft = win_len
    fre_used_ratio = 2000 / (sample_rate / 2)
    #fre_used_ratio = 4000 / (sample_rate / 2) # what is this...?
    # concentrate freq from 0~4000hz.
    # paper...



    # nbatch, nsample, nch
    # stft = DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio)

    nsample = signal.shape[2]
    nch = signal.shape[1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    nb = signal.shape[0]
    nt = int((nsample) / win_shift) + 1 # for iSTFT
    # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
    stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)

    window = torch.hann_window(window_length=win_len, device=signal.device)
    for ch_idx in range(0, nch, 1):
        stft_temp = torch.stft(signal[:, ch_idx, :], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                               window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

        stft[:, :, :, ch_idx] = stft_temp[:, 0:nf, 0:nt] #+ torch.finfo(torch.float32).eps)

    #stft = torch.real(stft)

    IID = torch.zeros((nb,nf,nt), dtype=torch.float32)
    for d_idx in range(nb):
        IID[d_idx,:,:] = 20*torch.log10(torch.abs(stft[d_idx,:,:,0])) - 20*torch.log10(torch.abs(stft[d_idx,:,:,1]))
        IID[d_idx,:,:] /= (torch.max(torch.abs(IID[d_idx,:,:]))+torch.finfo(torch.float32).eps)

    #IID = IID / torch.abs(IID).max() # not utilizing is better

    return IID.to(device)

def sigs2sinIPD(signal, device): # can consume cpu memory due to numpy usage, should be changed in long term

    sample_rate = 16000

    win_len_sec = 0.256 #1.024 #0.256
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5 #0.1 #0.5
    nfft = win_len
    fre_used_ratio = 2000 / (sample_rate / 2)
    #fre_used_ratio = 4000 / (sample_rate / 2) # what is this...?
    # concentrate freq from 0~4000hz.
    # paper...


    # nbatch, nsample, nch
    # stft = DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio)

    nsample = signal.shape[2]
    nch = signal.shape[1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    nb = signal.shape[0]
    nt = int((nsample) / win_shift) + 1 # for iSTFT
    # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
    stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)

    window = torch.hann_window(window_length=win_len, device=signal.device)
    for ch_idx in range(0, nch, 1):
        stft_temp = torch.stft(signal[:, ch_idx, :], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                               window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

        stft[:, :, :, ch_idx] = stft_temp[:, 0:nf, 0:nt] #+ torch.finfo(torch.float32).eps)

    #stft = torch.real(stft)

    IID = torch.sin(torch.angle(stft[:,:,:,0]/(stft[:,:,:,1]+ torch.finfo(torch.float32).eps)))

    #IID = IID / torch.abs(IID).max() # not utilizing is better

    return IID.to(device)


def sigs2IPD(signal, device): # can consume cpu memory due to numpy usage, should be changed in long term

    sample_rate = 16000

    win_len_sec = 0.256 #1.024 #0.256
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5 #0.1 #0.5
    nfft = win_len
    fre_used_ratio = 2000 / (sample_rate / 2)
    #fre_used_ratio = 4000 / (sample_rate / 2) # what is this...?
    # concentrate freq from 0~4000hz.
    # paper...


    # nbatch, nsample, nch
    # stft = DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio)

    nsample = signal.shape[2]
    nch = signal.shape[1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    nb = signal.shape[0]
    nt = int((nsample) / win_shift) + 1 # for iSTFT
    # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
    stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)

    window = torch.hann_window(window_length=win_len, device=signal.device)
    for ch_idx in range(0, nch, 1):
        stft_temp = torch.stft(signal[:, ch_idx, :], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                               window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

        stft[:, :, :, ch_idx] = stft_temp[:, 0:nf, 0:nt] #+ torch.finfo(torch.float32).eps)

    #stft = torch.real(stft)

    IID = torch.angle(stft[:,:,:,0]/(stft[:,:,:,1]+ torch.finfo(torch.float32).eps))

    #IID = IID / torch.abs(IID).max() # not utilizing is better

    return IID.to(device)

def sigs2sinIPDIID(signal, device): # can consume cpu memory due to numpy usage, should be changed in long term

    sample_rate = 16000

    win_len_sec = 0.256 #1.024 #0.256
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5 #0.1 #0.5
    nfft = win_len
    fre_used_ratio = 2000 / (sample_rate / 2)
    #fre_used_ratio = 4000 / (sample_rate / 2) # what is this...?
    # concentrate freq from 0~4000hz.
    # paper...


    # nbatch, nsample, nch
    # stft = DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio)

    nsample = signal.shape[2]
    nch = signal.shape[1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    nb = signal.shape[0]
    nt = int((nsample) / win_shift) + 1 # for iSTFT
    # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
    stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)

    window = torch.hann_window(window_length=win_len, device=signal.device)
    for ch_idx in range(0, nch, 1):
        stft_temp = torch.stft(signal[:, ch_idx, :], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                               window = window, center = True, normalized = False, return_complex = True)  # for iSTFT

        stft[:, :, :, ch_idx] = stft_temp[:, 0:nf, 0:nt] #+ torch.finfo(torch.float32).eps)

    #stft = torch.real(stft)

    IPD = torch.sin(torch.angle(stft[:,:,:,0]/(stft[:,:,:,1]+ torch.finfo(torch.float32).eps)))
    IID = torch.zeros((nb,nf,nt), dtype=torch.float32)
    for d_idx in range(nb):
        IID[d_idx,:,:] = 20*torch.log10(torch.abs(stft[d_idx,:,:,0])) - 20*torch.log10(torch.abs(stft[d_idx,:,:,1]))
        IID[d_idx,:,:] /= (torch.max(torch.abs(IID[d_idx,:,:]))+torch.finfo(torch.float32).eps)

    ans = torch.cat([IPD.unsqueeze(1), IID.unsqueeze(1)],1)

    #IID = IID / torch.abs(IID).max() # not utilizing is better

    return ans.to(device)

