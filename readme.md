# CTC_CE_for_AST

## Introduction

This is the source code of the following paper:

Jun-You Wang and Jyh-Shing Roger Jang, "Training a Singing Transcription Model Using Connectionist Temporal Classification Loss and Cross-entropy Loss," accepted by *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.

Using the source code in this repo, one should be able to reproduce the experiments discussed in the paper. Note that we did not fix the random number during model training, so the reproduced results may slightly deviate from the results reported in the paper. More resources (pretrained models, shifted MIR-ST500 dataset labels, results for plotting figures, etc) can be found [here](https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx?usp=sharing).

## Our contribution

In the paper, we introduce a method of using both cross-entropy loss and CTC loss to train a *note-level singing transcription* (hereinafter abbreviated as *singing transcription*) model.

This model accepts *monophonic vocal with instruments* audio as input (contains only one lead vocal and instruments), and outputs the note sequence that the vocalist sings, with quantized *note pitch* and unquantized *onset time* and *offset time* attributes for each note.

For more details, please refer to the paper itself. Basically, the most important feature is that our model can learn from ***weakly labeled data*** which contains only audio and *note sequence* labels, which will be discussed in "Reproduce the experiments" section.

## Quick start

It is possible to run all the source code on python 3.7 and 3.8. I didn't test the code on python 3.9 or above, but I think those environments should also be OK. 

### Install dependencies

The simplest way to install all dependencies is:

```
pip install -r requirements.txt
```

If this does not work due to some install order issues, then try this command:

```
cat requirements.txt | xargs -n 1 pip install
```

The above command will also install *mir_eval*, *matplotlib*, *madmom*, and other packages that are only required for model evaluation, plotting results, or other parts that are not related to inference. If you do not want to install those packages, just modify `requirements.txt` to remove these dependencies.

### Inference one song

To use the pretrained model for inference, run:

```
python do_everything.py [input_path] [output_path] \
    [model_path] [yaml_path] [device]
```

where `input_path` is the path to the audio file to be transcribed; `output_path` is the path to the MIDI file that the transcription will be written to; `model_path` is the path to the model checkpoint for singing transcription; `yaml_path` is the path to the config yaml file; `device` specifies the device used to perform singing transcription (cuda:0, cpu, etc).

As for pretrained model checkpoints, we have uploaded pretrained model checkpoints and the config files related to the checkpoints [here](https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx?usp=sharing)\*.

Take `ctc_ce#3_98` as an example, the corresponding config is `inference_ce_ctc#3_98.yaml`. Suppose the checkpoint is put under`models` directory, while the config file is put under `configs` directory, and the input audio name is `shenqing.wav`\(you can download this demo audio [here](https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx?usp=sharing)**), the desired output path is `shenqing.mid`, and the GPU resource is not available, then you can run:

```
python do_everything.py shenqing.wav shenqing.mid models/ctc_ce#3_98 \
 configs/inference_ce_ctc#3_98.yaml cpu
```

That's it. To perform singing transcription on other songs, just modify *input path* and *output path*.

(2023.01.15 updated) The result should be identical (or at least almost identical, due to different environment/version of python packages) to `shenqing_ctc_ce#3_98.mid`, which can also be downloaded from the same folder on google drive.

\* Note that the config files for different model checkpoints (even with the same model architecture) may be different because the *onset threshold* and *offset threshold* vary across model checkpoints, which are two hyper-parameters that should be tuned.

\** Thanks to 吳定洋 (the composer of this song) and 林大鈞 (the singer of this song) for granting the right of using this song for demo. The original video can be found on Youtube [here](https://www.youtube.com/watch?v=J8r6pMwFH2w).

## Reproduce experiments

This section describes how to reproduce all the experiments discussed in the paper, including model training, hyper-parameter search, model evaluation, global time shift correction (shift MIR-ST500's labels by +30ms), t-tests, and more.

### Prepare data

Put all the training data in a folder. The folder contains several subfolders, each of which contains the training data of a song, including one "vocal" file and one "instrument" file. The vocal/instrument files should have the same file name across the whole dataset, for example, "Vocal.wav" for vocal files, "Inst.wav" for instrument files. 

The prepared dataset would look like:

```
MIR-ST500 (dataset name)
    |
    ├－ 1 (song id)
    |    ├－ Vocal.wav (extracted vocal file name)
    |    └── Inst.wav (extracted accompaniments file name)
    |
    ├－ 2 (song id)
    |    ├－ Vocal.wav (extracted vocal file name)
    |    └── Inst.wav (extracted accompaniments file name)
    └── ...
```

In our experiments, the "Vocal.wav" and "Inst.wav" files are obtained by applying [Spleeter](https://github.com/deezer/spleeter) (2 stems) on the mixture for the MIR-ST500 dataset. As for the ISMIR2014 dataset, the "Vocal.wav" is the original audio (which does not contain any instrument), while the "Inst.wav" is silence.

The name of each subfolder represents the id of this specific training data, which should be the same as the id in the groundtruth note label file (JSON file).

### Feature extraction

Before training the model, first we extract features from audio and save them as hdf5 format. The config `feature_extraction/gen_feature.yaml` is an example, showing all the arguments required for feature extraction, which are listed as follows:

```
feature_extractor_file: "data_utils/get_feature.py"
feature_extractor_class_name: "CQT_feature_extractor"
audio_dir: "../MIR-ST500"
vocal_name: "Vocal.wav"
inst_name: "Inst.wav"
feature_output_name: "feature.hdf5"
```

where `feature_extractor_file` specifies the path to the python file which contains a feature extractor class, `feature_extractor_class_name` specifies the class name of the feature extractor. `audio_dir` specifies the path to the dataset, `vocal_name` and `inst_name` are the file name of the vocal files and instrument files, respectively. `feature_output_name` is the desired file name of the h5 files. These files will be written to the same directory as the audio files.

After defining all these arguments, run:

```
cd feature_extraction
python generate_feature.py [config_path]
```

where `config_path` is the path to the cofig yaml file.

### Create groundtruth data (frame-level & sequence-level)

After feature extraction, we still need to convert the groundtruth note labels to the frame-level and sequence-level labels that can be read by my main program. This step includes the weight assignment for each frame.

First, I have to clarify the difference between *strongly-labeled* and *weakly-labeled* dataset, because the python scripts for parsing the two kinds of data are different.

#### Strongly-labeled and weakly-labeled dataset

In our paper, we use three attributes to define a sung note, namely *onset*, *offset*\* and *note pitch*. *Onset* is the time when the note starts, *offset* is the time when the note ends, *note pitch* is the pitch value that the singer sings, which is quantized to semitones based on music theory (i.e., integer MIDI number).

We define *strongly-labeled* dataset as the dataset with note labels, where **all three attributes** of each note are well-annotated. On the other hand, the note labels of *weakly-labeled* dataset contains **only note pitch** labels of each note. 

The reason we have to discriminate the two kinds of dataset is that, in practice, most of the musical score you can find online can only be used as *weak labels*, because the musical performance does not always align with the musical score. Although such a deviation may be only 100ms (and sometimes is hard to be identified), it is still large enough to cause troubles in model training.

However, it is not that easy to create *strong labels*, because *onset* and *offset* labels have to be adjusted carefully to make them strictly align with the audio (basically, the error should be less than 50ms, or 0.05 second, which is very difficult to achieve). Therefore, it is desirable to use *weakly-labeled* dataset for model training.

Fortunately, our singing transcription model can be trained with either strongly-labeled or weakly-labeled data.

\*Some other papers may use *onset*, *duration*, and *note pitch* to define a note, which is similar to MIDI format. The *offset* can be obtained by adding *duration* to *onset*.

#### For strongly-labeled data

First, you should convert the note labels to a dictionary (with json format). Each key of the dictionary is the id of the song. The value of that entry is the note label of the corresponding song. The note label contains a list of notes, where each note is represented by a list of [onset, offset, note pitch]. Therefore, the note label of a song may look like: [[0.5, 1.0, 60], [1.5, 2.0, 62], [2.0, 2.33, 70], ......]. 

Note that we use the key of the dictionary to match the audio with the labels, so the key should be defined properly. 

Then, run the following command:

```
python gt_strong_dataset.py [gt_json_path] [output_path] \
    [dataset_dir] [feature_file_name] [weighting_method]
```

This script generates a pickle file from strong labels. The pickle file contains both the frame-level labels and sequence-level labels (i.e., the note sequence), which will then be used for model training.

Here, `gt_json_path` is the path to the json dictionary file, `output_path` specifies the path of the output pickle file, `dataset_dir` specifies the directory of the dataset (the folder described in the "Prepare data" section), `feature_file_name` specifies the name of h5 feature files (it needs this argument because it has to know the total frame numbers in order to generate frame-level labels). The last argument, `weighting_method`, specifies the method of assigning frame-level labels and weights to the onset frame classification subtask. Available methods are *CE* and *CE+CTC* (which is the same as *CE smooth* in the paper). Please refer to the paper for more details about these methods. 

#### For weakly-labeled data

Similar to strongly-labeled data, you should also convert the note labels to a dictionary (with json format). Each key of the dictionary is the id of the song. The value of that entry is the note label of the corresponding song. The note label contains a list of notes. Here, each note is only represented by a number of note pitch. **NO** onset or offset label is needed. The note label of a song may look like: [60, 60, 62, 70, ......].

Note that if there are consecutive notes with the same note pitch value, the value should still be repeated. The reason is that, a single note pitch number in this list represents *a note*, which is different from *two consecutive notes with the same note pitch*.

Then, run the following command:

```
python gt_weak_dataset.py [gt_json_path] [output_path] [dataset_dir]
```

This script generates a pickle file from weak labels. The pickle file contains only sequence-level labels (i.e., the note sequence).

The meaning of the three arguments here are the same as  `gt_strong_dataset.py`. Since there is no exact onset and offset label, we cannot create frame-level labels. Therefore, we don't need `weighting_method` anymore.

### Model training

We have to first write a training config file for model training. The config `config/train_config.yaml` is an example, showing all the arguments required for model training, which are listed as follows:

```
network_file: "net/onset_and_pitch_0901.py"
network_class_name: "Split_onset_pitch"

use_strongly_dataset: True
training_dataset_dir: "../MIR-ST500"
training_gt_pkl_label: "MIR-ST500_train_ctcce+30_1027.pkl"

use_weakly_dataset: True
training_weakly_dataset_dir: "../MIR-ST500"
training_weakly_gt_pkl_label: "MIR-ST500_train_weak_1027.pkl"

val_dataset_dir: "../val_set"
val_gt_pkl_label: "MIR-ST500_test_ctcce+30_1027.pkl"

h5py_feature_file_name: "feature.hdf5"

epoch: 100
warmup_epoch: 5
lr: 0.0001
batch_size: 1
model_save_dir: "models"
model_save_prefix: "ctc_ce_test"
log_path: "logger_ctc_ce_test.pkl"
```

`network_file` is the path to the network (model architecture) file, `network_class_name` is the class name of the neural network. 

`training_dataset_dir` is the directory of the **strongly-labeled** training dataset, `training_gt_pkl_label` is the path to the **strongly-labeled** pickle labels (generated by `gt_strong_dataset.py` from the strong note labels). `use_strongly_dataset` speficies whether the strongly-labeled dataset will be used. You would not have to specify `training_dataset_dir` and `training_gt_pkl_label` if `use_strongly_dataset` is set to `False`.

`training_weakly_dataset_dir` is the directory of the **weakly-labeled** training dataset, `training_weakly_gt_pkl_label` is the path to the **weakly-labeled** pickle labels (generated by `gt_weak_dataset.py` from the weak note labels). `use_weakly_dataset` speficies whether the weakly-labeled dataset will be used. You would not have to specify `training_weakly_dataset_dir` and `training_weakly_gt_pkl_label` if `use_weakly_dataset` is set to `False`.

Note that you have to set either `use_strongly_dataset` or `use_weakly_dataset` to `True` for obvious reason. For *CE* or *CE smooth* settings, `use_strongly_dataset` is `True` while `use_weakly_dataset` is `False`; for *CE+CTC* setting, both `use_strongly_dataset` and `use_weakly_dataset` are `True`; for *CTC* setting, `use_strongly_dataset` is `False` while `use_weakly_dataset` is `True`.

`val_dataset_dir`is the directory of the validation dataset, `val_gt_pkl_label` is the path to the pickle labels of the validation dataset. In my implementation, the validation dataset should always be a **strongly-labeled** dataset.

`h5py_feature_file_name` specifies the name of h5 feature files.

`epoch` specifies the number of epochs for training. We define an *epoch* as the number of iterations required to iterate the **weakly labeled training dataset** once. If the weakly labeled data is not used for training, then an epoch is defined based on strongly labeled training dataset. This definition affects extreme settings such as *CE10+CTC400* a lot, since 1000 iterations (10 strongly-labeled songs $\times$ 100) are not enough to train a decent singing transcription model.

`warmup_epoch` specifies the number of warmup epochs (where CTC loss is disabled), `lr` is the learning rate, `batch_size` is the batch size (to save GPU memory, the batch size of the dataloader is always fixed at 1, and this so-called batch size is achieved by calling optimizer.step() after multiple forward passes), `model_save_dir` is the directory to save model checkpoints, `model_save_prefix` is the prefix of model checkpoints (the model checkpoint name will be `model_save_prefix` followed by the '_' symbol and epoch number). `log_path` specifies the output path of the training log pickle file. 

Then, run the following command:

```
python train.py [yaml_path] [device] [--pretrained_path]
```

where `yaml_path` is the path of the training config file, `device` is the device used for training (cpu, cuda:0, etc), `--pretrained_path` is an optional argument that specifies the path to the pretrained model.

### Hyper-parameter exhaustive search

After model training, we have to search for the best model checkpoint, onset threshold and offset threshold. To perform hyper-parameter search, we need to write a config similar to `search_param/search_param_config.yaml`, which is shown as follows:

```
network_file: "../net/onset_and_pitch_0901.py"
network_class_name: "Split_onset_pitch"
dataset_dir: "../../MIR-ST500"
h5py_feature_file_name: "feature.hdf5"
gt_json_path: "../json/MIR-ST500_corrected_0514_+30ms.json"
epoch: 100
onset_thres_start: 0.02
onset_thres_end: 0.98
onset_thres_step_size: 0.02
offset_thres_start: 0.02
offset_thres_end: 0.98
offset_thres_step_size: 0.02
```

The meaning of `network_file`, `network_class_nam`, `h5py_feature_file_name` arguments are the same as the model training config. `dataset_dir` is the directory to the dataset used for hyper-parameter search. `gt_json_path` is the path to the groundtruth (json) files. The dataset used here must be **strongly-labeled** dataset because our criteria for searching parameters are related to COn and COnPOff F1-scores, both of which require strong groundtruth labels to compute.

Then, `epoch` is the total number of epochs for training, `onset_thres_start`, `onset_thres_end` define the minimum and maximum value of onset threshold, `onset_thres_step_size` is the step size for grid search. Similarly, `offset_thres_start`, `offset_thres_end` define the minimum and maximum value of offset threshold, `offset_thres_step_size` is the step size for grid search.

After defining these arguments, run the following command:

```
cd search_param
python predict_each_epoch.py [yaml_path] [device] \
    [prediction_output_prefix] [model_path_prefix]
```

`predict_each_epoch.py` loads every model checkpoints and inference the dataset. Instead of generating the transcription results, it dumps the models' frame-level prediction (onset probability, silence probability, etc) to pickle files. 

Here, `yaml_path` is the path to the config file, `device` is the device used for inference, `prediction_output_prefix` specifies the prefix of the path to dump the model predictions, `model_path_prefix` specifies the prefix of model checkpoints.

For example, if `prediction_output_prefix` is `raw_output/ctc_ce#1`, `model_path_prefix` is `models/ctc_ce#1`, and the number of epoch is 100, then this script will load `models/ctc_ce#1_1`, dump the prediction to `raw_output/ctc_ce#1_1`, and then load `models/ctc_ce#1_2`, dump the prediction to `raw_output/ctc_ce#1_2`  ......, until `models/ctc_ce#1_100`.

Then, we run the following command to automatically try different sets of parameters for post-processing, which will then determine the best set of parameters:

```
python find_parameter.py [yaml_path] [prediction_output_prefix] \
    [predicted_threshold_output_path] [model_performance_output_path]
```

The `find_parameter.py` script loads the models' frame-level prediction (onset probability, silence probability, etc) from pickle files, and then try different onset and offset thresholds to obtain the best hyper-parameters **for each model checkpoint**. After obtaining these hyper-parameters, it would then be trivial to find the best model checkpoint.

Again, `yaml_path` is the path to the config file, `prediction_output_prefix` specifies the prefix of the pickle files where the model predictions are. After an extremely time-consuming hyper-parameter searching process, the best onset and offset threshold for each model checkpoint will be written to `predicted_threshold_output_path`, and the model performance using these best thresholds will be written to  `model_performance_output_path`.

In our experiments reported in the paper, we used the whole MIR-ST500 training set for hyper-parameter searching. Under this setting, this step would take **a lot of** times. This is quite normal.

### Model testing

This script performs singing transcription on a whole dataset. The feature files should be prepared (using `generate_feature.py`) beforehand.

```
python predict.py [yaml_path] [dataset_dir] [h5py_feature_file_name] \
    [predict_json_file] [model_path] [device]
```

Similar to `do_everything.py`, `yaml_path` is the path to the config yaml file, `model_path` is the path to the model checkpoint, `device` specifies the device used for singing transcription.

Additionally, `dataset_dir` is the directory to the test dataset, `h5py_feature_file_name` specifies the name of h5 feature files, `predict_json_file` is the desired path where the output JSON file will be written to.

This python program reads a model checkpoint, and test the model on a whole dataset, and outputs a JSON file similar to the **strong labels** (as discussed in "Create groundtruth data" section).

Finally, to obtain the COn, COnP, and COnPOff F1-scores, run the following command:

```
python evaluate.py [gt_file] [predicted_file] [tol]
```

where `gt_file` is the path to the groundtruth note labels (JSON file, must be strong labels), `predicted_file` is the path to the predicted JSON file, `tol` is the onset tolerance, which is set to `0.05` (i.e., 50ms) in most of our experiments.

This python program will then print the model performance to standard output.

### Estimated runtime (as a reference)

Based on my (maybe somewhat inaccurate) memory, using an NVIDIA 1080 ti GPU and a Intel Core i9-7900X CPU, the estimated runtime is about:

- Inference (`do_everything.py`): **30 seconds** for each 4-minute song

- Prepare data (Run spleeter for SVS, w/ CPU): **1~2 hours** for the MIR-ST500 dataset.

- Feature extraction: **1 hour** for the MIR-ST500 dataset.

- Create groundtruth data: **less than 5 minutes** for the MIR-ST500 dataset.

- Model training (MIR-ST500 training set, 100 epochs, batch size=1, using GPU): **1 day** if only cross-entropy loss is used, **1.5~2 days** if both cross-entropy loss and CTC loss are used.

- Hyper-parameter exhaustive search (MIR-ST500 training set, 100 epochs, using GPU): **4 hours** to run `predict_each_epoch.py`, **1 day** to run `find_parameter.py`.

- Model testing (MIR-ST500 test set, 100 songs, using GPU): **less than 5 minutes**.

### Misc

Here are the commands used to reproduce other parts of experiments/figures in the paper. They are not directly related to model training, so I will not explain them in detail.

#### Quantify the MIR-ST500 labels time shift issue

Take the MIR-ST500 dataset as an example. For the ISMIR2014 dataset, you should first convert the TXT note labels to JSON format, and then run similar command.

Suppose the directory to the MIR-ST500 dataset's audio is `../../MIR-ST500`, the file name of extracted vocal is `Vocal.wav`, and the path to the MIR-ST500 dataset's groundtruth label is `../json/MIR-ST500_corrected_0514.json` (you can obtain it from [this repo](https://github.com/york135/singing_transcription_ICASSP2021)), then:

```
cd time_shift
python madmom_onset.py ../../MIR-ST500 Vocal.wav MIR-ST500_madmom.json
python compute_time_shift.py ../json/MIR-ST500_corrected_0514.json \
    MIR-ST500_madmom.json 0.05
```

The global time shift will then be printed to standard output. This should not take a lot of time. One hour should be enough to reproduce the results.

#### Plot figures from raw data

Here I will explain how to plot the figures shown in the paper. You can also modify these python scripts to print some of the raw experiment results which are not reported in the paper, such as the F1-scores of each trial (we repeated each model settings for 3~5 times, and reported only mean F1-scores and 95% confidence intervals). All the files mentioned below can be downloaded from [here](https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx?usp=sharing) (plotting_data folder).

**Figure 2.** The code is modified from [this repo](https://github.com/Itachi6912110/WAV2MIDI). First, put the shifted groundtruth (`MIR-ST500_corrected_0514_+30ms.json`) under the folder `json`, and put the song `468_Vocal.wav` (the extracted vocal of the song number 468 in MIR-ST500 dataset) under the folder `plotting`. Then, run the following command:

```
cd plotting
python figure2.py
```

It will then output `figure2.png`.

**Figure 3.** Put the pickle file `noshift#1_ismir2014_92_80_36_shift_plot.pkl`(the results of no time shift version of *CE*) and `ce#2_ismir2014_100_78_40_shift_plot.pkl` (the +30ms time shift version of *CE*), then run:

```
python figure3.py
```

It will then output `figure3.png`.

**Figure 4.** The evaluation results of *CE10, CE10+CTC10, CE10+CTC50, CE10+CTC100, CE10+CTC400* have been manually written to `figure4.py`. Therefore, simply run:

```
python figure4.py
```

It will then output `figure4.png`.

#### View T-test results

I have manually written all the experiment results (F1-scores for each trial) to the python files. Therefore, simply run:

```
cd t_tests
python strongly_data_ttest.py
python weakly_data_ttest.py
python ablation_ttest.py
```

will call scipy.stats.ttest_ind() function, and then display t-tests result.

#### Run Omnizart to obtain singing transcription results

Please refer to [Omnizart package](https://github.com/Music-and-Culture-Technology-Lab/omnizart). The only thing worth noting is that, to test the model performance of Omnizart on the ISMIR2014 dataset (*monophonic vocal*), we modified Omnizart's source code to disable the use of Spleeter (singing voice separation model), and directly passed the original audio to Omnizart's singing transcirption model (using *omnizart vocal transcribe* command). This would slightly affect the model performance, and we think disabling Spleeter should be reasonable because we do know that the ISMIR2014 dataset contains no instrument.

## Note

- In our paper, we propose to shift the groundtruth of the MIR-ST500 dataset by +30ms. The shifted groundtruth can be found at [here](https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx?usp=sharing), whose file name is `MIR-ST500_corrected_0514_+30ms.json`. The original groundtruth file can be found [here](https://github.com/york135/singing_transcription_ICASSP2021), whose file name is `MIR-ST500_corrected_0514.json`.

- Actually this is not the original source code I used in the experiments. The original source code is pretty messy, and most of the arguments have to be specified in the python files. I don't think this is a good coding style. Therefore, I reformulated the code and used yaml files to specify the arguments in the last couple of weeks. However, this may lead to unexpected bugs. Please feel free to open an issue if you spot any bug.

- If you have any song that can be used for demo (without any copyright issue), please let me know. I will appreciate it, and will try my best to promote your songs (maybe in this repo or other SNS pages).
