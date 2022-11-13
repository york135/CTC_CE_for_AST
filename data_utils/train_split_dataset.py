from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import concurrent.futures
import h5py
import time
import json
import pickle
import importlib, sys

class TrainSplitDataset(Dataset):

    def __init__(self):
        self.data_instances = {}
        self.instance_key = []
        self.instance_length = {}

        self.on_off_weight = None
        self.frame_level_gt_label = None
        self.gt_data_seq = None


    def init_feature_extractor(self, feature_extractor_file, feature_extractor_class_name):

        file_basename = os.path.basename(feature_extractor_file)
        mod_name = os.path.splitext(file_basename)[0]
        # print (file_basename, mod_name)

        spec = importlib.util.spec_from_file_location(mod_name, feature_extractor_file)
        feature_module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = feature_module

        spec.loader.exec_module(feature_module)
        Extractor = getattr(feature_module, feature_extractor_class_name)
        self.feature_extractor = Extractor()
        # print (self.feature_extractor)
        return

    def create_dataset(self, audio_dir, vocal_name, inst_name, feature_output_name):
        self.data_instances = []
        self.instance_key = []

        total_count = len(os.listdir(audio_dir))
        self.temp_cqt = {}
        future = {}
        print (time.time())

        for the_dir in tqdm(os.listdir(audio_dir)):
            
            vocal_path = os.path.join(audio_dir, the_dir, vocal_name)
            inst_path = os.path.join(audio_dir, the_dir, inst_name)

            feature_data = self.feature_extractor.get_all_feature(vocal_path, inst_path)
            feature_data = feature_data.permute(1, 0, 2)

            cur_output_path = os.path.join(audio_dir, the_dir, feature_output_name)

            with h5py.File(cur_output_path, "w") as f:
                f.create_dataset("data", data=feature_data)
                f.create_dataset("instance_length", data=feature_data.shape[1])


    def load_dataset_from_h5(self, dataset_dir, h5py_feature_file_name):
        self.h5py_path = []
        self.instance_key = []
        for the_dir in tqdm(os.listdir(dataset_dir)):
            self.h5py_path.append(os.path.join(dataset_dir, the_dir, h5py_feature_file_name))
            self.instance_key.append(the_dir)

        self.data_length = len(os.listdir(dataset_dir))

    def load_gt_pkl_label(self, gt_dataset_path):
        with open(gt_dataset_path, 'rb') as f:
            self.on_off_weight, self.frame_level_gt_label, self.gt_data_seq = pickle.load(f)

    def __getitem__(self, idx):
        cur_key = self.instance_key[idx]
        if not cur_key in self.data_instances.keys():
            self.data_instances[cur_key] = h5py.File(self.h5py_path[idx], 'r')["data"][()]
            self.instance_length[cur_key] = h5py.File(self.h5py_path[idx], 'r')["instance_length"][()]

        if self.gt_data_seq is None:
            # During testing, neither weak label nor strong label is available.
            return (self.data_instances[cur_key], cur_key)

        elif self.frame_level_gt_label is None:
            # Weakly labeled data
            # If the data has only weak labels, then batch[2] and batch[3] will be torch.zeros(1), which is simply a placeholder
            return (self.data_instances[cur_key], self.instance_length[cur_key]
                , torch.zeros(1), torch.zeros(1), self.gt_data_seq[cur_key], cur_key)
        else:
            # Strongly labeled data
            return (self.data_instances[cur_key], self.instance_length[cur_key]
                , self.on_off_weight[cur_key], self.frame_level_gt_label[cur_key]
                , self.gt_data_seq[cur_key], cur_key)


    def __len__(self):
        # return len(self.data_instances)
        return self.data_length


