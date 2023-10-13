import torch, h5py
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np

import sys, os, importlib
import math
import statistics

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)
# FRAME_LENGTH = 0.01

class CTC_CE_loss(nn.Module):
    def __init__(self, device):
        super(CTC_CE_loss, self).__init__();
        # Onset weight for CE: 32.54:1 (MIR-ST500 training set)
        self.onset_criterion = nn.BCELoss(reduction='none')
        self.offset_criterion = nn.BCELoss(reduction='none')
        self.on_off_criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.device = device

    def forward(self, batch, model_output, total_split_loss, use_ctc, use_ce):
        
        _, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, all_result = model_output

        onset_logits = on_off_logits_sm[:, :, 1]
        offset_logits = on_off_logits_sm[:, :, 2]

        # To avoid numirical instability 
        onset_logits = torch.clip(onset_logits, min=1e-7, max=1 - 1e-7)
        offset_logits = torch.clip(offset_logits, min=1e-7, max=1 - 1e-7)

        if use_ce == True:
            # Strong labels are needed. If the data has only weak labels, then batch[2] and batch[3] will be torch.zeros(1), which is simply a placeholder
            onset_prob = batch[3][:, :, 0].float().to(self.device)
            offset_prob = batch[3][:, :, 1].float().to(self.device)
            pitch_octave = batch[3][:, :, 2].long().to(self.device)
            pitch_class = batch[3][:, :, 3].long().to(self.device)

            on_weight = batch[2][0][0:1].float().to(self.device)
            off_weight = batch[2][0][1:2].float().to(self.device)

            split_train_loss0 = torch.dot(self.onset_criterion(onset_logits, onset_prob)[0], on_weight[0]) / torch.clip(torch.sum(on_weight), min=1e-10)
            split_train_loss1 = torch.dot(self.offset_criterion(offset_logits, offset_prob)[0], off_weight[0]) / torch.clip(torch.sum(off_weight), min=1e-10)
            split_train_loss2 = self.octave_criterion(pitch_octave_logits.permute(0, 2, 1), pitch_octave)
            split_train_loss3 = self.pitch_criterion(pitch_class_logits.permute(0, 2, 1), pitch_class)

        else:
            # No CE loss is used. Simply return zero tensors
            split_train_loss0 = torch.zeros((1,)).to(self.device)
            split_train_loss1 = torch.zeros((1,)).to(self.device)
            split_train_loss2 = torch.zeros((1,)).to(self.device)
            split_train_loss3 = torch.zeros((1,)).to(self.device)


        on_off_ctc_logits = all_result.permute(1, 0, 2)

        if use_ctc == True:
            on_off_seq = []
            # Define CTC loss groundtruth (maybe you can find a way to remove 1 and 2, while still involving onset and silence probability in loss function?)
            # Note that there is always a "silence" token (2) before the first note.
            # Each note is represented by an "onset" (1), a pitch value token (int(batch[4][0]) - 36 + 3), and a "silence" (2) token.
            # btw, 0 is blank.
            on_off_seq.append(2)
            for i in range(len(batch[4][0])):
                on_off_seq.append(1)
                on_off_seq.append(int(batch[4][0][i])-36+3)
                on_off_seq.append(2)

            on_off_seq = torch.tensor([on_off_seq,])

            on_off_seq = on_off_seq.to(self.device)

            split_train_loss4 = self.on_off_criterion(on_off_ctc_logits, on_off_seq
                , (on_off_ctc_logits.shape[0],), (on_off_seq.shape[1],))
        else:
            # No CTC loss is used. Simply return zero tensors
            split_train_loss4 = torch.zeros((1,)).to(self.device)
        

        total_split_loss[0] = total_split_loss[0] + split_train_loss0.item()
        total_split_loss[1] = total_split_loss[1] + split_train_loss1.item()
        total_split_loss[2] = total_split_loss[2] + split_train_loss2.item()
        total_split_loss[3] = total_split_loss[3] + split_train_loss3.item()
        total_split_loss[4] = total_split_loss[4] + split_train_loss4.item()

        # You can try to add weights to different loss terms here!
        loss = split_train_loss0 + split_train_loss1 + split_train_loss2 + split_train_loss3 + split_train_loss4
        return loss

class NoteLevelAST:
    def __init__(self, network_file, network_class_name, device= "cpu", model_path=None):

        self.device = device
        self.model = self.init_model(network_file, network_class_name).to(self.device)
        self.loss_function_class = CTC_CE_loss(self.device)
        
        if model_path is not None:
            self.load_model(model_path)
            
        print('Predictor initialized.')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location= self.device))
        print('Model read from {}.'.format(model_path))

    def init_model(self, network_file, network_class_name):

        file_basename = os.path.basename(network_file)
        mod_name = os.path.splitext(file_basename)[0]

        spec = importlib.util.spec_from_file_location(mod_name, network_file)
        feature_module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = feature_module

        spec.loader.exec_module(feature_module)
        Model_class = getattr(feature_module, network_class_name)
        model = Model_class()

        return model

    def fit(self, model_save_dir, **training_args):
        self.model_save_dir = model_save_dir
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)

        # Set training params
        self.batch_size = training_args['batch_size']
        self.epoch = training_args['epoch']
        self.lr = training_args['lr']
        self.save_every_epoch = training_args['save_every_epoch']
        self.log_path = training_args['log_path']

        self.use_weakly_dataset = training_args["use_weakly_dataset"]
        self.use_strongly_dataset = training_args["use_strongly_dataset"]

        self.warmup_epoch = training_args["warmup_epoch"]

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))
        
        # The "batch size" of the loaders are all fixed at 1. We achieve larger batch by calling optimizer.step() after several "batches".
        # We use this trick to avoid cuda OOM
        if self.use_strongly_dataset:
            self.train_loader = DataLoader(
                self.training_dataset,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
            )
            self.train_strongly_iterator = iter(self.train_loader)

        self.valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        if self.use_weakly_dataset:
            self.train_weakly_loader = DataLoader(
                self.training_weakly_dataset,
                batch_size=1,
                num_workers=4,
                pin_memory=False,
                shuffle=True,
                drop_last=True,
            )
            self.train_weakly_iterator = iter(self.train_weakly_loader)

        start_time = time.time()
        training_loss_list = []
        valid_loss_list = []
        split_loss_list = []
        valid_split_loss_list = []
        result_index_list = []

        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        print (self.iters_per_epoch)

        if self.use_weakly_dataset:
            epoch_step_num = len(self.train_weakly_loader)
        else:
            epoch_step_num = len(self.train_loader)


        print ("Number of warmup epoch", self.warmup_epoch)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epoch + 1):
            self.model.train()
            
            total_training_loss = 0
            total_split_loss = np.zeros(5)

            for batch_idx in tqdm(range(epoch_step_num)):
            # for batch_idx, ce_batch in tqdm(enumerate(self.train_loader)):

                # Strongly labeled data
                if self.use_strongly_dataset:
                    try:
                        ce_batch = next(self.train_strongly_iterator)
                    except:
                        self.train_strongly_iterator = iter(self.train_loader)
                        ce_batch = next(self.train_strongly_iterator)
                    
                    input_tensor = ce_batch[0].to(self.device)
                    # print (input_tensor.shape)
                    model_output = self.model(input_tensor)
                    loss = self.loss_function_class(ce_batch, model_output, total_split_loss, use_ctc=False, use_ce=True)
                    total_training_loss += loss.item()
                    loss = loss / float(self.batch_size)
                    loss.backward()


                # Weakly labeled data
                if self.use_weakly_dataset and epoch > self.warmup_epoch:
                    try:
                        ctc_batch = next(self.train_weakly_iterator)
                    except:
                        self.train_weakly_iterator = iter(self.train_weakly_loader)
                        ctc_batch = next(self.train_weakly_iterator)
                    
                    input_tensor = ctc_batch[0].to(self.device)
                    # print (input_tensor.shape)
                    model_output = self.model(input_tensor)
                    loss = self.loss_function_class(ctc_batch, model_output, total_split_loss, use_ctc=True, use_ce=False)
                    total_training_loss += loss.item()
                    loss = loss / float(self.batch_size)
                    loss.backward()

                # Clip gradient to avoid gradient exploding
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5.0)

                # Use this method to achieve large batch size while avoiding cuda OOM 
                if batch_idx % self.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()


            if epoch % self.save_every_epoch == 0:
                # Perform validation on strongly labeled data
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    split_val_loss = np.zeros(5)
                    for batch_idx, batch in enumerate(self.valid_loader):

                        input_tensor = batch[0].to(self.device)

                        model_output = self.model(input_tensor)
                        loss = self.loss_function_class(batch, model_output, split_val_loss, use_ctc=True, use_ce=True)
                        total_valid_loss += loss.item()

                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_save_dir) / (training_args['model_save_prefix']+'_{}'.format(epoch))
                torch.save(save_dict, target_model_path)

                # Save loss list
                # training_loss_list.append((epoch, total_training_loss/len(self.train_loader)))
                training_loss_list.append((epoch, total_training_loss/len(self.train_loader)))
                valid_loss_list.append((epoch, total_valid_loss/len(self.valid_loader)))

                total_split_loss[0] = total_split_loss[0] / len(self.train_loader)
                total_split_loss[1] = total_split_loss[1] / len(self.train_loader)
                total_split_loss[2] = total_split_loss[2] / len(self.train_loader)
                total_split_loss[3] = total_split_loss[3] / len(self.train_loader)
                total_split_loss[4] = total_split_loss[4] / len(self.train_loader)

                split_loss_list.append((epoch, total_split_loss))
                valid_split_loss_list.append((epoch, split_val_loss/len(self.valid_loader)))

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        training_loss_list[-1][1],
                        valid_loss_list[-1][1],
                        time.time()-start_time))

                print('split train loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} on_off_ctc {:.4f}'.format(
                        total_split_loss[0],
                        total_split_loss[1],
                        total_split_loss[2],
                        total_split_loss[3],
                        total_split_loss[4]
                    )
                )
                print('split val loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} on_off_ctc {:.4f}'.format(
                        split_val_loss[0]/len(self.valid_loader),
                        split_val_loss[1]/len(self.valid_loader),
                        split_val_loss[2]/len(self.valid_loader),
                        split_val_loss[3]/len(self.valid_loader),
                        split_val_loss[4]/len(self.valid_loader)
                    )
                )

        # Save loss to file
        with open(self.log_path, 'wb') as f:
            pickle.dump({'train': training_loss_list, 'valid': valid_loss_list, 'train_split':split_loss_list, 'valid_split':valid_split_loss_list
                , 'result_index': result_index_list}, f)

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def _parse_frame_info(self, frame_info, onset_thres, offset_thres):
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

    def predict(self, test_dataset, results_dict, show_tqdm, onset_thres, offset_thres, return_pitch_logit=False):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 1
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}

            if show_tqdm == True:
                print('Forwarding model...')
                for batch_idx, batch in enumerate(tqdm(test_loader)):
                    input_tensor = batch[0].to(self.device)
                    song_ids = batch[1]

                    result_tuple = self.model(input_tensor)
                    on_off_logits = result_tuple[1]
                    pitch_octave_logits = result_tuple[2]
                    pitch_class_logits = result_tuple[3]

                    onset_logits = on_off_logits[:, :, 1]
                    offset_logits = on_off_logits[:, :, 2]

                    onset_probs, offset_probs = (onset_logits.cpu().numpy(), offset_logits.cpu().numpy())
                    pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

                    # print (song_ids)
                    # Collect frames for corresponding songs
                    for bid, song_id in enumerate(song_ids):
                        for i in range(len(onset_probs[bid])):
                            
                            if return_pitch_logit:
                                frame_info = (onset_probs[bid][i], offset_probs[bid][i], torch.argmax(pitch_octave_logits[bid][i]).item()
                                    , torch.argmax(pitch_class_logits[bid][i]).item(), F.softmax(pitch_octave_logits[bid][i], dim=0).numpy()
                                    , F.softmax(pitch_class_logits[bid][i], dim=0).numpy())
                            else:
                                frame_info = (onset_probs[bid][i], offset_probs[bid][i], torch.argmax(pitch_octave_logits[bid][i]).item()
                                    , torch.argmax(pitch_class_logits[bid][i]).item())

                            song_frames_table.setdefault(song_id, [])
                            song_frames_table[song_id].append(frame_info)

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results_dict[song_id] = self._parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
                
        return results_dict, song_frames_table
