from madmom.audio.filters import LogarithmicFilterbank
from madmom.features.onsets import CNNOnsetProcessor, SpectralOnsetProcessor
from madmom.audio.signal import normalize
import scipy
from madmom.audio.signal import Signal
import madmom

import time, sys, os, json
import librosa

import numpy as np

def get_onset(wav_path):
    
    y, sr = librosa.core.load(wav_path, sr= None)
    sos = scipy.signal.butter(25, 100, btype= 'highpass', fs= sr, output='sos')
    wav_data= scipy.signal.sosfilt(sos, y)

    if sr != 44100:
        wav_data = librosa.core.resample(wav_data, orig_sr= sr, target_sr= 44100)
        sr= 44100


    wav_data = librosa.util.normalize(wav_data)

    sodf = CNNOnsetProcessor()
    
    onset_strength = (sodf(Signal(data=wav_data, sample_rate=sr)))

    frame_per_sec = 100.0

    peaks = madmom.features.onsets.peak_picking(onset_strength, threshold=0.5, smooth=None, pre_avg=0, post_avg=0, pre_max=10, post_max=10)

    peaks = peaks / frame_per_sec
    # print (peaks)
    print ("Final onset len: {}".format(len(peaks)))
    
    return peaks

if __name__ == '__main__':

    dataset_dir = sys.argv[1]
    wav_name = sys.argv[2]


    transcription_result = {}

    for wav_dir in os.listdir(dataset_dir):
        wav_path = os.path.join(dataset_dir, wav_dir, wav_name)
        print (wav_path, "start processing time: %f" %(time.time()))

        onset_times = get_onset(wav_path)

        cur_transcription = []
        for i in range(len(onset_times)):
            cur_transcription.append([onset_times[i], onset_times[i] + 0.1, 60])

        transcription_result[wav_dir] = list(cur_transcription)


    with open(sys.argv[3], 'w') as f:
        output_string = json.dumps(transcription_result)
        f.write(output_string)