import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import argparse
import json, sys, os
import yaml
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluate import MirEval


FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)
song_frames_table = None


def parse_frame_info(frame_info, onset_thres, offset_thres):
    """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""

    result = []
    current_onset = None

    pitch_counter = []

    last_onset = 0.0
    onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
    
    local_max_size = 3
    current_time = 0.0

    onset_seq_length = len(onset_seq)

    for i in range(len(frame_info)):

        current_time = FRAME_LENGTH*i
        info = frame_info[i]

        backward_frames = i - local_max_size
        if backward_frames < 0:
            backward_frames = 0

        forward_frames = i + local_max_size + 1
        if forward_frames > onset_seq_length - 1:
            forward_frames = onset_seq_length - 1

        if info[0] >= onset_thres and (i - backward_frames) == np.argmax(onset_seq[backward_frames : forward_frames]):

            if current_onset is None:
                current_onset = current_time
                last_onset = info[0] - onset_thres

            else:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    
                current_onset = current_time
                last_onset = info[0] - onset_thres
                pitch_counter = []

        elif info[1] >= offset_thres:  # If is offset (silence prob > offset threshold)
            if current_onset is not None:
                if len(pitch_counter) > 0:
                    result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                current_onset = None
                
                pitch_counter = []

        # If current_onset exist, add count for the pitch
        if current_onset is not None:
            final_pitch = int(info[2]* 12 + info[3])
            if info[2] != 4 and info[3] != 12:
                # pitch_counter[final_pitch] += 1
                pitch_counter.append(final_pitch)

    if current_onset is not None:
        if len(pitch_counter) > 0:
            result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

        current_onset = None
        pitch_counter = []

    return result


def post_processing(onset_thres, offset_thres, song_ids):
    global song_frames_table
    results = {}
    # print ("===")
    # print (time.time())
    for song_id in song_ids:
        results[str(song_id)] = parse_frame_info(song_frames_table[song_id], onset_thres=onset_thres, offset_thres=offset_thres)

    # print (time.time())
    return results

def evaluate_ast(gt_path, ast_result):
    eval_class = MirEval()
    eval_class.add_gt(gt_path)
    eval_class.add_tr_tuple_and_prepare(ast_result)
    eval_result = eval_class.accuracy(onset_tolerance=0.05, print_result=False)

    return eval_result


def main(args):

    global song_frames_table

    with open(args.yaml_path, 'r') as stream:
        select_param_args = yaml.load(stream, Loader=yaml.FullLoader)

    best_on_off_thres_of_each_epoch = []
    # best_offset_thres_of_each_epoch = []

    print ("start searching onset threshold......", time.time())
    result_index_list = []

    start_epoch = 1
    end_epoch = select_param_args["epoch"] + 1

    for i in range(start_epoch, end_epoch):
        raw_file_path = args.prediction_output_prefix + "_" + str(i) + ".pkl"
        with open(raw_file_path, 'rb') as f:
            song_frames_table = pickle.load(f)

        onset_thres_start = select_param_args["onset_thres_start"]
        onset_thres_end = select_param_args["onset_thres_end"]
        onset_thres_set = [float(onset_thres_start),]

        while onset_thres_set[-1] < onset_thres_end:
            onset_thres_set.append(select_param_args["onset_thres_start"] + select_param_args["onset_thres_step_size"] * len(onset_thres_set))

        # print (onset_thres_set)

        future = [[] for k in range(len(onset_thres_set))]
        results = [[] for k in range(len(onset_thres_set))]
        song_ids = list(song_frames_table.keys())

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for k in range(len(onset_thres_set)):
                future[k] = executor.submit(post_processing, onset_thres_set[k], 0.5, song_ids)

            for k in range(len(onset_thres_set)):
                results[k] = future[k].result()

        # print (time.time())
        best_con = 0.0
        best_result = None
        best_con_thres = None

        eval_results = [[] for k in range(len(onset_thres_set))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for k in range(len(onset_thres_set)):
                future[k] = executor.submit(evaluate_ast, select_param_args["gt_json_path"], results[k])

            for k in range(len(onset_thres_set)):
                eval_results[k] = future[k].result()

        for k in range(len(onset_thres_set)):
            eval_result = eval_results[k]
            if eval_result[8] >= best_con:
                best_con = eval_result[8]
                best_result = list(eval_result)
                best_con_thres = onset_thres_set[k]

        print ("start searching offset threshold......", time.time())

        # Find best offset parameter
        offset_thres_start = select_param_args["offset_thres_start"]
        offset_thres_end = select_param_args["offset_thres_end"]
        offset_thres_set = [float(offset_thres_start),]

        while offset_thres_set[-1] < offset_thres_end:
            offset_thres_set.append(select_param_args["offset_thres_start"] + select_param_args["offset_thres_step_size"] * len(offset_thres_set))

        # print (offset_thres_set)

        future = [[] for k in range(len(offset_thres_set))]
        results = [[] for k in range(len(offset_thres_set))]
        song_ids = list(song_frames_table.keys())

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for k in range(len(offset_thres_set)):
                future[k] = executor.submit(post_processing, best_con_thres, offset_thres_set[k], song_ids)

            for k in range(len(offset_thres_set)):
                results[k] = future[k].result()

        best_conpoff = 0.0
        best_result = None
        best_offset_thres = None
        eval_results = [[] for k in range(len(offset_thres_set))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for k in range(len(offset_thres_set)):
                future[k] = executor.submit(evaluate_ast, select_param_args["gt_json_path"], results[k])

            for k in range(len(offset_thres_set)):
                eval_results[k] = future[k].result()

        for k in range(len(offset_thres_set)):
            eval_result = eval_results[k]
            if eval_result[2] >= best_conpoff:
                best_conpoff = eval_result[2]
                best_result = list(eval_result)
                best_offset_thres = offset_thres_set[k]

        best_on_off_thres_of_each_epoch.append([best_con_thres, best_offset_thres])

        print ("epoch", i, "onset threshold =", best_con_thres, "offset threshold =", best_offset_thres)
        print("         Precision Recall F1-score")
        print("COnPOff  %f %f %f" % (best_result[0], best_result[1], best_result[2]))
        print("COnP     %f %f %f" % (best_result[3], best_result[4], best_result[5]))
        print("COn      %f %f %f" % (best_result[6], best_result[7], best_result[8]))
        print ("gt note num:", best_result[9], "tr note num:", best_result[10])

        print ("epoch", i, "completed,", time.time())
        
        result_index_list.append([i, best_result])

    with open(args.predicted_threshold_output_path, 'w') as f:
        output_string = json.dumps(best_on_off_thres_of_each_epoch)
        f.write(output_string)

    with open(args.model_performance_output_path, 'wb') as f:
        pickle.dump(result_index_list, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path')
    parser.add_argument('prediction_output_prefix')
    parser.add_argument('predicted_threshold_output_path')
    parser.add_argument('model_performance_output_path')

    args = parser.parse_args()
    main(args)