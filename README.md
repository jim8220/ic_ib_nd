# Inter-channel information-based noise detector

![fig1](https://github.com/jim8220/ic_ib_nd/assets/68427972/cc192122-db30-4c8b-a06e-4fe142e92381)

# Dataset preparation
Download ToyADMOS2 https://github.com/nttcslab/ToyADMOS2-dataset

Run mixer.py

After running, ./processed_data will be made

# Noise detection

Basically, you modify baseline.yaml (or infer2others.yaml) and run python code.

Each variable in yaml file has below meaning and options.

<baseline.yaml>

machine_type: toy car / toy train


machine type that you want to experiment with

bg: 1 / 2 / 3 / 4


background noise that you want to experiment with

rseed: (recommended) 0 / 10 / 20


random seed that you want to experiment with

model_type: intensity_and_phase_spectrum / IID / IPD / sinIPD / IID+sinIPD


input feature that you want to experiment with


the reason why name is model_type is each model is defined separately for each input feature

==========================================================================

<infer2others.yaml>

bg_from : 1 / 2 / 3 / 4


background noise that is used for train and validation

bg_to : 1 / 2 / 3 / 4 (different from bg_from)


background noise that will be tested in

==========================================================================

other parameters

batch_size : (recommended) 16 

batch size


short_cut : (recommended) True

you can pass data loading process if you make this True and data is already made as 'dataset_prepared'


epochs : (recommended) 20 

epochs


exclude : (recommended) [aL, aM, aH, bL, bM, bH, cL, cM, cH, dL, dM, dH] 

machine condition that will be excluded during train and validation


## on-site
Modify machine_type, bg, rseed, model_type of baseline.yaml and run noise_detection.py

results can be found in ./dataset_prepared/.../result/

## other site
Modify machine_type, bg_from, bg_to, rseed, model_type of infer2others.yaml and run noise_detection_infer2others.py

results can be found in ./infer2others/.../result/

### draw
Modify machine_type, bg, rseed, model_type of baseline.yaml and run draw.py

If you have any questions, please email to lasscap@kaist.ac.kr


@inproceedings{harada2021toyadmos2,
    author = "Harada, Noboru and Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Yasuda, Masahiro and Saito, Shoichiro",
    title = "{ToyADMOS2}: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions",
    booktitle = "Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)",
    address = "Barcelona, Spain",
    month = "November",
    year = "2021",
    pages = "1--5",
    isbn = "978-84-09-36072-7",
    doi. = "10.5281/zenodo.5770113",
    _pdf = {https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf}
}
