import librosa
import os
import numpy as np
import concurrent.futures
import torch

class CQT_feature_extractor():
    def __init__(self):
        return

    def get_cqt(self, y, filter_scale=1):
        return np.abs(librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(24)
            , n_bins=96*4, bins_per_octave=12*4, filter_scale=filter_scale)).T

    def get_feature(self, y, sr):
        # y = librosa.util.normalize(y)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future1_1 = executor.submit(self.get_cqt, y, 0.5)
            future1_2 = executor.submit(self.get_cqt, y, 1.0)
            future1_3 = executor.submit(self.get_cqt, y, 2.0)

            cqt_feature1_1 = future1_1.result()
            cqt_feature1_2 = future1_2.result()
            cqt_feature1_3 = future1_3.result()


            cqt_feature1_1 = torch.tensor(cqt_feature1_1, dtype=torch.float).unsqueeze(1)
            cqt_feature1_2 = torch.tensor(cqt_feature1_2, dtype=torch.float).unsqueeze(1)
            cqt_feature1_3 = torch.tensor(cqt_feature1_3, dtype=torch.float).unsqueeze(1)
            cqt_feature1 = torch.cat((cqt_feature1_1, cqt_feature1_2, cqt_feature1_3), dim=1)

        return cqt_feature1

    def get_all_feature_from_array(self, y_voc, y_mix):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future2_1 = executor.submit(self.get_cqt, y_voc, 0.5)
            future2_2 = executor.submit(self.get_cqt, y_voc, 1.0)
            future2_3 = executor.submit(self.get_cqt, y_voc, 2.0)
            future1_1 = executor.submit(self.get_cqt, y_mix, 0.5)
            future1_2 = executor.submit(self.get_cqt, y_mix, 1.0)
            future1_3 = executor.submit(self.get_cqt, y_mix, 2.0)

            cqt_feature2_1 = future2_1.result()
            cqt_feature2_2 = future2_2.result()
            cqt_feature2_3 = future2_3.result()
            cqt_feature1_1 = future1_1.result()
            cqt_feature1_2 = future1_2.result()
            cqt_feature1_3 = future1_3.result()

            cqt_feature2_1 = torch.tensor(cqt_feature2_1, dtype=torch.float).unsqueeze(1)
            cqt_feature2_2 = torch.tensor(cqt_feature2_2, dtype=torch.float).unsqueeze(1)
            cqt_feature2_3 = torch.tensor(cqt_feature2_3, dtype=torch.float).unsqueeze(1)

            cqt_feature2 = torch.cat((cqt_feature2_1, cqt_feature2_2, cqt_feature2_3), dim=1)

            cqt_feature1_1 = torch.tensor(cqt_feature1_1, dtype=torch.float).unsqueeze(1)
            cqt_feature1_2 = torch.tensor(cqt_feature1_2, dtype=torch.float).unsqueeze(1)
            cqt_feature1_3 = torch.tensor(cqt_feature1_3, dtype=torch.float).unsqueeze(1)

            cqt_feature1 = torch.cat((cqt_feature1_1, cqt_feature1_2, cqt_feature1_3), dim=1)

        cqt_data = torch.cat((cqt_feature1, cqt_feature2), dim=1)
        return cqt_data

    def get_all_feature(self, voc_path, acc_path, voc_isfile=True, acc_isfile=True):

        if voc_isfile == True:
            y, sr = librosa.core.load(voc_path, sr=None, mono=True)
            if sr != 44100:
                y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
        else:
            # voc_path is vocal part, while acc_path is instrument part.
            y = voc_path
        
        if acc_isfile == True:
            y2, sr2 = librosa.core.load(acc_path, sr=None, mono=True)
            if sr2 != 44100:
                y2 = librosa.core.resample(y= y2, orig_sr= sr2, target_sr= 44100)
        else:
            # voc_path is vocal part, while acc_path is instrument part.
            y2 = acc_path
            
        y_voc = y
        y_mix = np.add(y, y2)

        max_mag = np.max(np.abs(y_mix))

        y_voc = y_voc / (max_mag+0.0001)
        y_mix = y_mix / (max_mag+0.0001)

        return self.get_all_feature_from_array(y_voc, y_mix).permute(1, 0, 2)


    def compute_svs(self, y, sr):
        from spleeter.separator import Separator
        import warnings
        separator = Separator('spleeter:2stems')
        warnings.filterwarnings('ignore')

        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)

        waveform = np.expand_dims(y, axis= 1)

        prediction = separator.separate(waveform)
        # print (prediction["vocals"].shape)
        ret_voc = librosa.core.to_mono(prediction["vocals"].T)
        ret_voc = np.clip(ret_voc, -1.0, 1.0)

        ret_acc = librosa.core.to_mono(prediction["accompaniment"].T)
        ret_acc = np.clip(ret_acc, -1.0, 1.0)
        del separator

        return ret_voc, ret_acc

    def get_feature_from_mixture(self, wav_path, do_svs):

        y, sr = librosa.core.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
        y = librosa.util.normalize(y)

        if do_svs == True:
            y_voc, y_acc = self.compute_svs(y, 44100)

            max_mag = np.max(np.abs(np.add(y_voc, y_acc)))
            y_voc = y_voc / (max_mag+0.0001)
            y_acc = y_acc / (max_mag+0.0001)

            y_mix = np.add(y_voc, y_acc)
            cqt_data = self.get_all_feature_from_array(y_voc, y_mix)
        else:
            cqt_data = self.get_all_feature_from_array(y, np.zeros(y.shape))

        return cqt_data.permute(1, 0, 2)
