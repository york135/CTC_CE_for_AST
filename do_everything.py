import sys, os, json
import time
import argparse
import torch
import yaml

from predictor import NoteLevelAST
from data_utils.seq_dataset import SeqDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from tqdm import tqdm
import mido
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length
    return mid
    

def convert_to_midi(predicted_result, song_id, output_path):
    to_convert = predicted_result[song_id]
    mid = notes2mid(to_convert)
    mid.save(output_path)

def predict_one_song(predictor, inference_kwargs, wav_path, song_id, results_dict, midi_output_path=None, raw_output_path=None, return_pitch_logit=False):
    test_dataset = SeqDataset()
    test_dataset.init_feature_extractor(inference_kwargs["feature_extractor_file"], inference_kwargs["feature_extractor_class_name"])
    test_dataset.create_dataset_for_one_song(wav_path, song_id, do_svs=inference_kwargs["do_svs"])

    results_dict, song_frames_table = predictor.predict(test_dataset, results_dict=results_dict, show_tqdm=inference_kwargs["show_tqdm"]
        , onset_thres=inference_kwargs["onset_thres"], offset_thres=inference_kwargs["offset_thres"]
        , return_pitch_logit=return_pitch_logit)

    if raw_output_path is not None:
        with open(raw_output_path, 'wb') as f:
            pickle.dump(song_frames_table, f, protocol=4)

    if inference_kwargs["tomidi"] == True and midi_output_path is not None:
        convert_to_midi(results_dict, song_id, midi_output_path)

    return results_dict, song_frames_table


def main(args):
    # model_path = args.model_path
    wav_path = args.input_path
    output_path = args.output_path

    with open(args.yaml_path, 'r') as stream:
        inference_kwargs = yaml.load(stream, Loader=yaml.FullLoader)

    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    # print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    predictor = NoteLevelAST(network_file=inference_kwargs["network_file"], network_class_name=inference_kwargs["network_class_name"], device=device, model_path=args.model_path)
    song_id = '1'
    results = {}
    predict_one_song(predictor, inference_kwargs, wav_path, song_id, results, midi_output_path=output_path)


if __name__ == '__main__':
    print (time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('model_path')
    parser.add_argument('yaml_path')
    parser.add_argument('device')

    args = parser.parse_args()

    main(args)
    print (time.time())
