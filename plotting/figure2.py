import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import math, json, sys, os, argparse, time, pickle
import numpy as np
import sys

import pretty_midi
import librosa
import librosa.display

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from spleeter.separator import Separator
separator = Separator('spleeter:2stems')

def get_vocal(y, sr):

    if sr != 44100:
        y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)

    waveform = np.expand_dims(y, axis= 1)

    prediction = separator.separate(waveform)
    # print (prediction["vocals"].shape)
    ret_voc = librosa.core.to_mono(prediction["vocals"].T)
    ret_voc = np.clip(ret_voc, -1.0, 1.0)

    # ret_acc = librosa.core.to_mono(prediction["accompaniment"].T)
    # ret_acc = np.clip(ret_acc, -1.0, 1.0)

    return ret_voc


def plot_curve(mix_path, ce_ctc_note_seq, ce_ctc_onset_seq, ce_ctc_silence_seq, ce_ctc_octave_seq, ce_ctc_chroma_seq, cur_song_gt, plot_path, Start_T=88.0, End_T=100.0):
    
    ce_ctc_chroma_seq = np.transpose(ce_ctc_chroma_seq)
    ce_ctc_octave_seq = np.transpose(ce_ctc_octave_seq)

    print (ce_ctc_onset_seq.shape, ce_ctc_silence_seq.shape, ce_ctc_octave_seq.shape, ce_ctc_chroma_seq.shape)
    y, sr = librosa.core.load(mix_path, sr=44100, mono=True)
    vocal = get_vocal(y, sr)

    hop_length = 1024

    voc_chroma = librosa.feature.chroma_cqt(y=vocal, sr=sr, hop_length=hop_length)

    frame_size_sec = 1024.0 / 44100.0
    fig, (ax_chroma_orig, ax_ce_ctc_onset, ax_ce_ctc_silence, ax_ce_ctc_octave, ax_ce_ctc_chroma, ax) = plt.subplots(6, 1, sharex = True
                                                    , gridspec_kw = {'height_ratios':[2, 1, 1, 1, 2, 2]}
                                                    , figsize=(10, 15))


    x_label = np.array([float(i) * frame_size_sec for i in range(len(ce_ctc_onset_seq))])

    print (voc_chroma.shape)
    # voc_chroma = voc_chroma[:, int(Start_T/frame_size_sec):int(End_T/frame_size_sec)]
    # print (voc_chroma.shape)
    librosa.display.specshow(voc_chroma[:,:int(End_T/frame_size_sec)], y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length, ax=ax_chroma_orig)

    librosa.display.specshow(ce_ctc_octave_seq[:,:int(End_T/frame_size_sec)], x_axis='time', sr=sr, hop_length=hop_length, ax=ax_ce_ctc_octave)
    ax_ce_ctc_octave.set_yticks([0, 1, 2, 3, 4])
    ax_ce_ctc_octave.set_yticklabels([2, 3, 4, 5, 'N'])

    librosa.display.specshow(ce_ctc_chroma_seq[:,:int(End_T/frame_size_sec)], y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length, ax=ax_ce_ctc_chroma)
    ax_ce_ctc_chroma.set_yticks([0, 2, 4, 5, 7, 9, 11, 12])
    ax_ce_ctc_chroma.set_yticklabels(['C', 'D', 'E', 'F', 'G', 'A', 'B', 'N'])
    
    t_contour, = ax_ce_ctc_onset.plot(x_label[int(Start_T/frame_size_sec):int(End_T/frame_size_sec)]
        , ce_ctc_onset_seq[int(Start_T/frame_size_sec):int(End_T/frame_size_sec)], 'b-', label = 'Onset contour')
    t_off_contour, = ax_ce_ctc_silence.plot(x_label[int(Start_T/frame_size_sec):int(End_T/frame_size_sec)]
        , ce_ctc_silence_seq[int(Start_T/frame_size_sec):int(End_T/frame_size_sec)], 'r-', label = 'Silence contour')
    # t_off_contour, = ax4.plot(x_label[int(Start_T//frame_size):int(End_T//frame_size)], silence_seq[int(Start_T//frame_size):int(End_T//frame_size)], 'k-', label = 'CE+CTC silence contour')

    ax_chroma_orig.set(xlabel='', ylabel='Chromagram')
    
    ax_ce_ctc_onset.set(xlabel='', ylabel='Onset prob')
    ax_ce_ctc_silence.set(xlabel='', ylabel='Silence prob')
    ax_ce_ctc_chroma.set(xlabel='', ylabel='Pitch class')
    ax_ce_ctc_octave.set(xlabel='', ylabel='Octave')

    ax.set(xlabel='time (s)', ylabel='Note pitch (MIDI number)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # ax_chroma_orig.set(xlim=(Start_T, End_T),)
    # ax_ce_onset.axis([Start_T, End_T, 0.0, 1.1])
    ax_ce_ctc_onset.axis([Start_T, End_T, 0.0, 1.1])
    # ax3.axis([Start_T, End_T, 0.0, 1.1])
    # ax4.axis([Start_T, End_T, 0.0, 1.1])

    # #fig, ax = plt.subplots()
    # ax.imshow(Z_linear, aspect='auto', cmap='Purples', \
    #                origin='lower', extent=[Start_T, End_T, Start_F, End_F])

    # pitch_contour, = ax.plot(x_filt, y_filt, 'b--', label = 'Pitch contour')

    mksize = 6
    effective_pitch = []

    for i in range(ce_ctc_note_seq.shape[0]):
        if ce_ctc_note_seq[i][1] >= Start_T and ce_ctc_note_seq[i][0] <= End_T:
            x_est = [ce_ctc_note_seq[i][0], ce_ctc_note_seq[i][1]]
            y_est = [ce_ctc_note_seq[i][2] + 0.05, ce_ctc_note_seq[i][2] + 0.05]
            interval2, = ax.plot(x_est, y_est, 'g-', label = 'Note prediction', linewidth=5.0, alpha=0.5)
            on2, = ax.plot(x_est[0], y_est[0], 'go', label = 'Onsets', markersize=mksize, alpha=0.7)
            # off2, = ax.plot(x_est[-1], y_est[-1], 'yx', label = 'CE+CTC offset', markersize=mksize)

            effective_pitch.append(ce_ctc_note_seq[i][2])

    for i in range(cur_song_gt.shape[0]):
        if cur_song_gt[i][1] >= Start_T and cur_song_gt[i][0] <= End_T:
            x_est = [cur_song_gt[i][0], cur_song_gt[i][1]]
            y_est = [cur_song_gt[i][2], cur_song_gt[i][2]]
            interval3, = ax.plot(x_est, y_est, '-', label = 'Groundtruth', linewidth=5.0, alpha=0.3, color='grey')
            on3, = ax.plot(x_est[0], y_est[0], 'o', label = 'Groundtruth onset', markersize=mksize, color='grey', alpha=0.5)
            # off3, = ax.plot(x_est[-1], y_est[-1], 'gx', label = 'Groundtruth offset', markersize=mksize)

            effective_pitch.append(cur_song_gt[i][2])

    min_pitch = min(effective_pitch)
    max_pitch = max(effective_pitch)

    ax.axis([Start_T, End_T, min_pitch - 6, max_pitch + 1])


    #ax.tick_params(labelsize=14)
    #plt.xlabel("t (s)", fontsize=16)
    #plt.ylabel("f (Hz)", fontsize=16)
    plt.grid(True, axis='y', alpha=0.7, linestyle='-.')
    plt.legend(handles=[interval2, interval3], fontsize=8, loc='lower left')
    #plt.legend(handles=[pitch_contour, interval1, interval3], fontsize=8, loc='lower left')
    #plt.tight_layout()
    # plt.show(fig)
    fig.savefig(plot_path, dpi=500)


def read_midi(midi_path):
    mid_file = pretty_midi.PrettyMIDI(midi_path)
    # print (mid_file)
    tuples = []
    for instrument in mid_file.instruments:
        # print (instrument)
        for note in instrument.notes:
            tuples.append([note.start, note.end, note.pitch])

    tuples = np.array(tuples)
    # print (tuples)
    return tuples

if __name__ == '__main__':
    song_id = '468'

    ce_ctc_raw_file_path = '468_ctc_ce#1_100.pkl'
    midi_path = '468_ctc_ce#1_100_28_70.mid'
    

    ce_ctc_note_seq = read_midi(midi_path)

    with open(ce_ctc_raw_file_path, 'rb') as f:
        ce_ctc_song_frames_table = pickle.load(f)

    ce_ctc_frame_info = ce_ctc_song_frames_table[song_id]

    ce_ctc_onset_seq = np.array([ce_ctc_frame_info[i][0] for i in range(len(ce_ctc_frame_info))])
    # ce_ctc_silence_seq = np.array([ce_ctc_frame_info[i][1] for i in range(len(ce_ctc_frame_info))])
    ce_ctc_silence_seq = np.array([ce_ctc_frame_info[i][1] for i in range(len(ce_ctc_frame_info))])
    ce_ctc_octave_seq = np.array([ce_ctc_frame_info[i][4] for i in range(len(ce_ctc_frame_info))])
    ce_ctc_chroma_seq = np.array([ce_ctc_frame_info[i][5] for i in range(len(ce_ctc_frame_info))])


    gt_path = '../json/MIR-ST500_corrected_0514_+30ms.json'
    with open(gt_path) as json_data:
        gt = json.load(json_data)

    cur_song_gt = np.array(gt[song_id])

    plot_path = 'figure2.png'
    mix_path = '468_Vocal.wav'
    plot_curve(mix_path, ce_ctc_note_seq, ce_ctc_onset_seq, ce_ctc_silence_seq, ce_ctc_octave_seq, ce_ctc_chroma_seq, cur_song_gt, plot_path)

