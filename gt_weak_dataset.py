import os, shutil
import numpy as np
import h5py

from tqdm import tqdm

import librosa
import time
import json
import torch

import argparse
import pickle

def generate_frame_level_groundtruth_labels(dataset_dir, gt_json_path, output_path, weighting_method="CE_CTC"):
    # generate the labels
    # We still need dataset_dir, because some songs in the gt_json_path may not be used for training.
    # We have to identify the songs that are really used for training, and only use them to form the dataset.

    with open(gt_json_path) as json_data:
        gt = json.load(json_data)

    gt_data_seq = {}

    for song_dir in tqdm(os.listdir(dataset_dir)):

        gt_data = np.array(gt[song_dir])

        if len(gt_data.shape) == 2 and gt_data.shape[1] == 3:
            # Treat strongly labeled data (the form of [[onset, offset, pitch], ......]) as weakly labeled data
            gt_data_seq[song_dir] = gt_data[:,2]
        else:
            # For the "real" weakly labeled data ([pitch, pitch, pitch, ......], without onset and offset labels)
            gt_data_seq[song_dir] = gt_data

    with open(output_path, 'wb') as f:
        # Two placeholder. Replace "on_off_weight_label" and "frame_level_gt_label" in the strongly labeled dataset.
        pickle.dump([None, None, gt_data_seq], f, protocol=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_json_path')
    parser.add_argument('output_path')
    parser.add_argument('dataset_dir')

    args = parser.parse_args()

    generate_frame_level_groundtruth_labels(args.dataset_dir, args.gt_json_path, args.output_path)