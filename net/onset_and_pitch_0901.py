import torch.nn as nn
import torch
import torch.nn.functional as F
import time

class Onset_cnn(nn.Module):
    def __init__(self):
        super(Onset_cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(9, 9), stride=(1, 4), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            )

        self.fc1   = nn.Sequential(
            nn.Linear(32*(96*8)//8, 64),
            nn.ReLU(),
            )
        self.fc2   = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            )
        self.fc3   = nn.Sequential(
            nn.Linear(32, 4),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = torch.flatten(out.permute(0, 2, 3, 1), start_dim=2) 
        out = self.fc1(out)
        out = self.fc2(out)
        on_off_logits = self.fc3(out)

        return on_off_logits

class Pitch_cnn(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(Pitch_cnn, self).__init__()
        self.pitch_octave = pitch_octave
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(9, 9), stride=(1, 4), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            )

        self.fc1   = nn.Sequential(
            nn.Linear(32*(96*8)//8, 64),
            nn.ReLU(),
            )
        self.fc2   = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            )
        self.fc3   = nn.Sequential(
            nn.Linear(32, pitch_class+pitch_octave+2),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = torch.flatten(out.permute(0, 2, 3, 1), start_dim=2) 
        out = self.fc1(out)
        out = self.fc2(out)
        pitch_out = self.fc3(out)

        pitch_octave_logits = pitch_out[:, :, 0:(self.pitch_octave+1)]
        pitch_class_logits = pitch_out[:, :, (self.pitch_octave+1):]

        return pitch_octave_logits, pitch_class_logits



class Split_onset_pitch(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4, svs_path=None):
        super(Split_onset_pitch, self).__init__()

        self.pitch_octave = pitch_octave

        self.onset_cnn = Onset_cnn()
        self.pitch_cnn = Pitch_cnn(pitch_class=pitch_class, pitch_octave=pitch_octave)
        
    def forward(self, x):
        
        on_off_logits = self.onset_cnn(x)
        pitch_octave_logits, pitch_class_logits = self.pitch_cnn(x)

        # 0: blank, 1:onset, 2:offset, 3:have pitch
        # The following operations are normalization operations. We use log softmax to avoid numerical instability.
        on_off_logits_sm = F.softmax(on_off_logits, dim=2)
        on_off_logits_log_sm = F.log_softmax(on_off_logits, dim=2)

        pitch_octave_sm = F.log_softmax(pitch_octave_logits[:,:,:self.pitch_octave], dim=2)
        pitch_class_sm = F.log_softmax(pitch_class_logits[:,:,:12], dim=2)

        # For CTC loss (normalize all tokens)
        all_result = torch.zeros((pitch_class_logits.shape[0], pitch_class_logits.shape[1], 3+4*12))

        all_result[:,:,0:3] = on_off_logits_log_sm[:,:,0:3]
        for i in range(4):
            for j in range(12):
                index_num = i*12+j+3
                all_result[:,:,index_num] = pitch_octave_sm[:,:,i] + pitch_class_sm[:,:,j] + on_off_logits_log_sm[:,:,3]

        return on_off_logits, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, all_result

if __name__ == '__main__':
    from torchsummary import summary
    model = Split_onset_pitch().cuda()
    summary(model, input_size=(2, 11, 84))
