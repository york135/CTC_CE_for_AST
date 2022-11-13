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


frame_size = 1024.0 / 44100.0
def preprocess(gt_data, length, pitch_shift=0):

    frame_label = []

    cur_note = 0
    cur_note_onset = gt_data[cur_note][0]
    cur_note_offset = gt_data[cur_note][1]
    cur_note_pitch = gt_data[cur_note][2] + pitch_shift

    if np.max(cur_note_pitch) > 83 or np.min(cur_note_pitch) < 36:
        return None, False

    # start from C2 (36) to B5 (83), total: 4 classes. This is a little confusing
    octave_start = 0
    octave_end = 3
    pitch_class_num = 12

    for i in range(length):
        cur_time = i * frame_size

        if abs(cur_time - cur_note_onset) <= (frame_size / 2.0 + 0.0001):
            # First dim : onset
            # Second dim : no pitch
            if i == 0 or frame_label[-1][0] != 1:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [1, 0, my_oct, my_pitch_class]
                # print (cur_time)
                frame_label.append(label)
            else:
                my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                my_pitch_class = cur_note_pitch % pitch_class_num
                label = [0, 0, my_oct, my_pitch_class]
                frame_label.append(label)

        elif cur_time < cur_note_onset or cur_note >= len(gt_data):
            # For the frame that doesn't belong to any note
            label = [0, 1, octave_end+1, pitch_class_num]
            frame_label.append(label)

        elif abs(cur_time - cur_note_offset) <= (frame_size / 2.0 + 0.0001):
            # For the offset frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num
            label = [0, 1, my_oct, my_pitch_class]

            cur_note = cur_note + 1
            if cur_note < len(gt_data):
                # print (gt_data[cur_note])
                cur_note_onset = gt_data[cur_note][0]
                cur_note_offset = gt_data[cur_note][1]
                cur_note_pitch = gt_data[cur_note][2] + pitch_shift
                if abs(cur_time - cur_note_onset)  <= (frame_size / 2.0 + 0.0001):
                    my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
                    my_pitch_class = cur_note_pitch % pitch_class_num
                    # print (cur_time)
                    label[0] = 1
                    label[1] = 0
                    label[2] = my_oct
                    label[3] = my_pitch_class
                    frame_label[-1][1] = 1

            frame_label.append(label)

        else:
            # For the voiced frame
            my_oct = int(min(max(octave_start, (cur_note_pitch- 36)//pitch_class_num), octave_end)) - octave_start
            my_pitch_class = cur_note_pitch % pitch_class_num

            label = [0, 0, my_oct, my_pitch_class]
            frame_label.append(label)

    # If the diffeerence frame
    cur_left = 0
    cur_right = 0
    onset_positive_range = 0.05

    for i in range(length):
        cur_time = i * frame_size

        while cur_time > gt_data[cur_right][0] and cur_right < len(gt_data) - 1:
            cur_right = cur_right + 1
            cur_left = cur_right - 1

        if abs(cur_time - gt_data[cur_left][0]) <= onset_positive_range or abs(cur_time - gt_data[cur_right][0]) <= onset_positive_range:
            frame_label[i].append(1)
        else:
            frame_label[i].append(0)
        
    return np.array(frame_label), True

def assign_ce_onset_weight(is_positive, distance):
    # For CE
    # 131618 onsets, 4283085 frames (1:~32.542) for MIR-ST500 training set
    if is_positive:
        return 32.54
    else:
        return 1.0

def assign_ce_ctc_onset_weight(is_positive, distance):
    # For CE+CTC or CE smooth
    if is_positive:
        return min(max(1.0-distance*20.0, 0.0), 1.0) * 20.0
    else:
        return 1.0

def generate_frame_level_groundtruth_labels(dataset_dir, gt_json_path, output_path, feature_file_name, weighting_method="CE_CTC"):
    # generate the labels and weights for each frame
    with open(gt_json_path) as json_data:
        gt = json.load(json_data)

    on_off_weight_label = {}
    frame_level_gt_label = {}
    gt_data_seq = {}

    onsets = 0
    silences = 0
    total_frame = 0

    note_num = 0
    answer_onset_num = 0
    song_num = 0

    for song_dir in tqdm(os.listdir(dataset_dir)):

        gt_data = np.array(gt[song_dir])

        # Need to access the feature file to know the number of frames for each song
        feature_path = os.path.join(dataset_dir, song_dir, feature_file_name)
        cur_inst_length = h5py.File(feature_path, 'r')["instance_length"][()]

        on_off_weight = torch.zeros((2, cur_inst_length))
        on_off_answer = torch.full((2, cur_inst_length), 1)
        cur_on = 0
        cur_off = 0
        
        frame_label, flag = preprocess(gt_data, cur_inst_length)

        note_num = note_num + len(gt_data)
        answer_onset_num = answer_onset_num + np.sum(frame_label[:,0])
        song_num = song_num + 1

        # If there is any note with pitch > 83 or < 36 (outside the pitch range of the singing transcription model), this condition is triggered.
        # If you want to deal with those notes with extremely high/low pitch values, you should modify the model architecture first.
        if flag == False:
            print (song_dir)
            continue

        for i in range(cur_inst_length):
            cur_t = i * frame_size

            total_frame = total_frame + 1

            # Find the distance from current frame to the nearest onset/offset
            if cur_t > gt_data[cur_on][0] and cur_on + 1 < len(gt_data):
                cur_on = cur_on + 1

            if cur_t > gt_data[cur_off][1] and cur_off + 1 < len(gt_data):
                cur_off = cur_off + 1

            onset_id = cur_on - 1
            onset_dist = abs(cur_t - gt_data[cur_on-1][0])
            if onset_dist > abs(cur_t - gt_data[cur_on][0]):
                onset_dist = abs(cur_t - gt_data[cur_on][0])
                onset_id = cur_on

            offset_id = cur_off - 1
            offset_dist = abs(cur_t - gt_data[cur_off-1][1])
            if offset_dist > abs(cur_t - gt_data[cur_off][1]):
                offset_dist = abs(cur_t - gt_data[cur_off][1])
                offset_id = cur_off

            if weighting_method == "CE":
                on_off_weight[0][i] = assign_ce_onset_weight(frame_label[i][0], onset_dist)
                on_off_answer[0][i] = frame_label[i][0]
                onsets = onsets + frame_label[i][0]

            if weighting_method == "CE_CTC":
                on_off_weight[0][i] = assign_ce_ctc_onset_weight(frame_label[i][4], onset_dist)
                on_off_answer[0][i] = frame_label[i][4]
                onsets = onsets + frame_label[i][0]

            on_off_weight[1][i] = 1.0
            on_off_answer[1][i] = frame_label[i][1]

            if frame_label[i][1] == 1:
                silences = silences + 1
            

        # print (list(on_off_weight[0]))
        # print (list(on_off_weight[1]))
        # print (list(on_off_answer[0]))
        # print (list(on_off_answer[1]))

        on_off_weight_label[song_dir] = on_off_weight
        frame_level_gt_label[song_dir] = frame_label
        gt_data_seq[song_dir] = gt_data[:,2]

    print (onsets, silences, total_frame)
    print (note_num, answer_onset_num, song_num)

    with open(output_path, 'wb') as f:
        pickle.dump([on_off_weight_label, frame_level_gt_label, gt_data_seq], f, protocol=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_json_path')
    parser.add_argument('output_path')
    parser.add_argument('dataset_dir')
    parser.add_argument('feature_file_name')
    parser.add_argument('weighting_method')

    args = parser.parse_args()

    generate_frame_level_groundtruth_labels(args.dataset_dir, args.gt_json_path, args.output_path, args.feature_file_name, args.weighting_method)