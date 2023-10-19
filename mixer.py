import librosa
from librosa.core.audio import __audioread_load
import soundfile as sf
import random
import numpy as np
import pickle
import os

f_down = 16000

anomaly_cond = ['aL','aM','aH','bL','bM','bH','cL','cM','cH','dL','dM','dH']


def mix(machine_path, noise_path, snr_db, dest_file, only_noise=False):

    y_machine, f_machine = __audioread_load(machine_path, offset=0.0, duration=None, dtype=np.float32)
    y_noise, f_noise = __audioread_load(noise_path, offset=0.0, duration=None, dtype=np.float32)


    y_machine = librosa.to_mono(y_machine) # for making sure
    y_machine = librosa.resample(y_machine, f_machine, f_down)

    y_noise = librosa.to_mono(y_noise) # for making sure
    y_noise = librosa.resample(y_noise, f_noise, f_down)


    machine_rms = np.sqrt(np.mean(y_machine**2))
    noise_rms = np.sqrt(np.mean(y_noise**2))


    k = machine_rms / noise_rms / 10**(snr_db / 20.)

    if only_noise == False:

        mixed = (y_machine * (1. / (1. + k))) + (y_noise * (k / (1. + k)))

    elif only_noise == True:

        mixed = y_noise * (k / (1. + k))


    finalized = (mixed * 32767.0).astype(np.int16)

    sf.write(dest_file, finalized, f_down, 'PCM_16')

def name2cond(name):
    return name.split('_')[0].split('-')[-2] + name.split('_')[0].split('-')[-1][-1]


# main

# upload file_list_noise.pkl

# upload file_list_machine_sound.pkl

# shuffle with fixed random seed

# for each pair, mix sound with random SNR sampled from uniform distribution -10 to 0
# list is like below

# =======noise=======
# Toy car / Toy train
# N1 / N2 / N3 / N4
# index
# mic
# ===================

# ======machine======
# Toy car / Toy train
# normal / anomaly
# index
# mic
# ===================

#np.random.seed(1)

# you can see index and mic part is in common.
# so peak pair randomly and mix it.

with open('file_list_noise.pkl', 'rb') as f_noise:
    pure_noise_list = pickle.load(f_noise)

f_noise.close()

with open('file_list_machine_sound.pkl', 'rb') as f_machine:
    pure_machine_list = pickle.load(f_machine)

f_machine.close()

# here, we will only divide N1 / N2 / N3 /N4
# train / valid / test will be separated in DL code, and will be managed by random seed (can be noted with .txt files, but by this way, various pairs can be tested)

# =====Toy car=====

machine_toycar = []
noise_toycar = []

# for each noises
print("Toy car")
for ionsite in range(4): #range(4) #noise 4 does not work.

    os.makedirs(f'./processed_data/ToyCar/N{ionsite + 1}', exist_ok=True)
    print(f"normal / onsite N{ionsite + 1}")
    # number of indexes

    nums = len(pure_noise_list[0][ionsite])

    nums_noise = nums // 2
    nums_machine = nums // 2
    nums_normal = nums_machine * 2 // 14
    nums_anomaly = nums_machine // 14

    # normal ===============================

    for inormal in range(nums_normal):
        imachine_idx = np.random.randint(len(pure_machine_list[0][0]))
        imachine = pure_machine_list[0][0][imachine_idx]
        del pure_machine_list[0][0][imachine_idx]

        inoise_idx = np.random.randint(len(pure_noise_list[0][ionsite]))
        inoise = pure_noise_list[0][ionsite][inoise_idx] #modify here
        del pure_noise_list[0][ionsite][inoise_idx]

        imachine_full_path = ['./data/ToyCar/normal/' + ipath for ipath in imachine]
        inoise_full_path = ['./data/ToyCar/env_noise/' + ipath for ipath in inoise]

        # mix !

        mixed_snr = np.random.uniform(-10,0)

        for imic in range(2):

            audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_SNR_{mixed_snr}.wav'
            mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyCar/N{ionsite+1}/{audio_name}', False)



    # anomaly ==============================

    # do same job, but for every anomalies

    print(f"anomaly / onsite N{ionsite + 1}")
    # number of indexes


    # have to separate each anomaly
    for ianomaly in anomaly_cond:
        print(ianomaly)
        anomalies2cond = np.array([name2cond(ifile[0]) for ifile in pure_machine_list[0][1]])
        specific_anomaly_list = np.array(pure_machine_list[0][1])[np.where(anomalies2cond == ianomaly)].tolist()

        for ianomalyidx in range(nums_anomaly):
            imachine_idx = np.random.randint(len(specific_anomaly_list))
            imachine = specific_anomaly_list[imachine_idx]
            del specific_anomaly_list[imachine_idx] # can be overlapped

            # to avoid overlap, we have to delete above instance directly from pure_machine_list

            del pure_machine_list[0][1][np.where(np.array(pure_machine_list[0][1]) == imachine)[0][0]]

            inoise_idx = np.random.randint(len(pure_noise_list[0][ionsite]))
            inoise = pure_noise_list[0][ionsite][inoise_idx]
            del pure_noise_list[0][ionsite][inoise_idx]

            imachine_full_path = ['./data/ToyCar/anomaly/' + ipath for ipath in imachine]
            inoise_full_path = ['./data/ToyCar/env_noise/' + ipath for ipath in inoise]

            # mix !

            mixed_snr = np.random.uniform(-10,0)

            for imic in range(2):

                audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_SNR_{mixed_snr}.wav'
                mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyCar/N{ionsite+1}/{audio_name}', False)

    # noise-only period ====================

    print(f"nosie-only period / onsite N{ionsite + 1}")
    # number of indexes

    for inormal in range(nums_noise):
        imachine_idx = np.random.randint(len(pure_machine_list[0][0]))
        imachine = pure_machine_list[0][0][imachine_idx]
        del pure_machine_list[0][0][imachine_idx]

        inoise_idx = np.random.randint(len(pure_noise_list[0][ionsite]))
        inoise = pure_noise_list[0][ionsite][inoise_idx]
        del pure_noise_list[0][ionsite][inoise_idx]

        imachine_full_path = ['./data/ToyCar/normal/' + ipath for ipath in imachine]
        inoise_full_path = ['./data/ToyCar/env_noise/' + ipath for ipath in inoise]

        # mix !

        mixed_snr = np.random.uniform(-10,0)

        for imic in range(2):

            audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_noise_only_period_SNR_{mixed_snr}.wav'
            mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyCar/N{ionsite+1}/{audio_name}', True)


    # do same job but with only_noise to True


# ====Toy Train====

machine_toytrain = []
noise_toytrain = []

# for each noises
print("Toy train")
for ionsite in range(0): # modify here by 0,1,2,3

    os.makedirs(f'./processed_data/ToyTrain/N{ionsite + 1}', exist_ok=True)
    print(f"normal / onsite N{ionsite + 1}")
    # number of indexes

    nums = len(pure_noise_list[1][ionsite])

    nums_noise = nums // 2
    nums_machine = nums // 2
    nums_normal = nums_machine * 2 // 14
    nums_anomaly = nums_machine // 14

    # normal ===============================

    for inormal in range(nums_normal):
        imachine_idx = np.random.randint(len(pure_machine_list[1][0]))
        imachine = pure_machine_list[1][0][imachine_idx]
        del pure_machine_list[1][0][imachine_idx]

        inoise_idx = np.random.randint(len(pure_noise_list[1][ionsite]))
        inoise = pure_noise_list[1][ionsite][inoise_idx]
        del pure_noise_list[1][ionsite][inoise_idx]

        imachine_full_path = ['./data/ToyTrain/normal/' + ipath for ipath in imachine]
        inoise_full_path = ['./data/ToyTrain/env_noise/' + ipath for ipath in inoise]

        # mix !

        mixed_snr = np.random.uniform(-10,0)

        for imic in range(4):

            audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_SNR_{mixed_snr}.wav'
            mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyTrain/N{ionsite+1}/{audio_name}', False)



    # anomaly ==============================

    # do same job, but for every anomalies

    print(f"anomaly / onsite N{ionsite + 1}")
    # number of indexes


    # have to separate each anomaly
    for ianomaly in anomaly_cond:
        print(ianomaly)
        anomalies2cond = np.array([name2cond(ifile[0]) for ifile in pure_machine_list[1][1]])
        specific_anomaly_list = np.array(pure_machine_list[1][1])[np.where(anomalies2cond == ianomaly)].tolist()

        for ianomalyidx in range(min(nums_anomaly, sum(anomalies2cond == ianomaly))): # modify here to avoid error
            imachine_idx = np.random.randint(len(specific_anomaly_list))
            imachine = specific_anomaly_list[imachine_idx]
            del specific_anomaly_list[imachine_idx] # can be overlapped

            # to avoid overlap, we have to delete above instance directly from pure_machine_list

            del pure_machine_list[1][1][np.where(np.array(pure_machine_list[1][1]) == imachine)[0][0]]

            inoise_idx = np.random.randint(len(pure_noise_list[1][ionsite]))
            inoise = pure_noise_list[1][ionsite][inoise_idx]
            del pure_noise_list[1][ionsite][inoise_idx]

            imachine_full_path = ['./data/ToyTrain/anomaly/' + ipath for ipath in imachine]
            inoise_full_path = ['./data/ToyTrain/env_noise/' + ipath for ipath in inoise]

            # mix !

            mixed_snr = np.random.uniform(-10,0)

            for imic in range(4):

                audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_SNR_{mixed_snr}.wav'
                mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyTrain/N{ionsite+1}/{audio_name}', False)

    # noise-only period ====================

    print(f"nosie-only period / onsite N{ionsite + 1}")
    # number of indexes

    for inormal in range(nums_noise):
        imachine_idx = np.random.randint(len(pure_machine_list[1][0]))
        imachine = pure_machine_list[1][0][imachine_idx]
        del pure_machine_list[1][0][imachine_idx]

        inoise_idx = np.random.randint(len(pure_noise_list[1][ionsite]))
        inoise = pure_noise_list[1][ionsite][inoise_idx]
        del pure_noise_list[1][ionsite][inoise_idx]

        imachine_full_path = ['./data/ToyTrain/normal/' + ipath for ipath in imachine]
        inoise_full_path = ['./data/ToyTrain/env_noise/' + ipath for ipath in inoise]

        # mix !

        mixed_snr = np.random.uniform(-10,0)

        for imic in range(4):

            audio_name = imachine[imic].split('.')[0] + inoise[imic].split('.')[0] + f'_noise_only_period_SNR_{mixed_snr}.wav'
            mix(imachine_full_path[imic], inoise_full_path[imic],mixed_snr, f'./processed_data/ToyTrain/N{ionsite+1}/{audio_name}', True)


    # do same job but with only_noise to True

