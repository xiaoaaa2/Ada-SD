import numpy as np
import os
import pandas as pd
import pdb
import random
import soundfile as sf
import torch
import torchaudio

import features
import torch.nn.functional as F
import time 

from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model
from typing import Union
from utils import AngProtoLoss4
#from pytorch_metric_learning.losses import SphereFaceLoss


class Voxceleb12_sample_cssl2(Dataset):

    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both Vox 1 + 2
    """

    def __init__(self, base_path1, base_path2, data_df1, data_df2, seed=0, window_len=32000):
        self.base_path1 = base_path1
        self.base_path2 = base_path2
        self.rel_path = pd.concat([data_df1['rel_path'], data_df2['rel_path']]).reset_index(drop=True)
        self.speaker = pd.concat([data_df1['speaker_id'], data_df2['speaker_id']]).reset_index(drop=True)
        self.len1 = len(data_df1)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        rpath = self.rel_path[index]
        if index < self.len1:
            abs_path = os.path.join(self.base_path1, rpath)
        else:
            abs_path = os.path.join(self.base_path2, rpath)

        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames

        max_start = length_arr - self.window_len
        start_ind = random.randint(0, max_start)
        seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)

        # for hard pos example
        curr_speaker = self.speaker[index]
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])

        rpath_pos = self.rel_path[pos_pair_ind]
        if pos_pair_ind < self.len1:
            abs_path_pos = os.path.join(self.base_path1, rpath_pos)
        else:
            abs_path_pos = os.path.join(self.base_path2, rpath_pos)
        # read in small chunk
        speech_arr_pos = sf.SoundFile(abs_path_pos)
        length_arr_pos = speech_arr_pos.frames

        max_start_pos = length_arr_pos - self.window_len
        start_ind_pos = random.randint(0, max_start_pos)
        seg_2, _ = sf.read(abs_path_pos, start=start_ind_pos, stop=start_ind_pos+self.window_len)
        
        seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)

        return seg_1, seg_2, curr_speaker


class Cnceleb12_sample_cssl(Dataset):
    
    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both CN 12
    """

    def __init__(self, base_path1, base_path2, data_df1, data_df2, seed=0, window_len=24000):
        self.base_path1 = base_path1
        self.base_path2 = base_path2
        self.rel_path = pd.concat([data_df1['rel_path'], data_df2['rel_path']]).reset_index(drop=True)
        self.speaker = pd.concat([data_df1['speaker_id'], data_df2['speaker_id']]).reset_index(drop=True)
        self.len1 = len(data_df1)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        self.speaker = list(self.speaker)
        
        #unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        # print('index', index)
        
        rpath = self.rel_path[index]
        # abs_path = os.path.join(self.base_path, rpath)
        abs_path = rpath
        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames
        if length_arr >= self.window_len:
            max_start = length_arr - self.window_len
            start_ind = random.randint(0, max_start)
            seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)
        else:
            seg_1, _ = sf.read(abs_path)

        # for hard pos example
        curr_speaker = self.speaker[index]
        neg_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp != curr_speaker])
        rpath_neg = self.rel_path[neg_pair_ind]
        abs_path_neg = rpath_neg
        speech_arr_neg = sf.SoundFile(abs_path_neg)
        # print('abs_path_neg', abs_path_neg)
        length_arr_neg = speech_arr_neg.frames
        # print('length_arr_neg', length_arr_neg)

        if length_arr_neg >= 10000:
            max_start_neg = length_arr_neg - 9000
            start_ind_neg = random.randint(0, max_start_neg)
            duration = random.randint(160, 8000)
            # print('duration, start_ind_neg', duration, start_ind_neg)
            seg_1_neg, _ = sf.read(abs_path_neg, start=start_ind_neg, stop=start_ind_neg+duration)
        else:
            seg_1_neg, _ = sf.read(abs_path_neg)  
       
        # print('length_arr_neg', len(seg_1_neg))        
        # print('seg_1_neg', seg_1_neg)    
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        # pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])

        # rpath_pos = self.rel_path[pos_pair_ind]
        # abs_path_pos = rpath_pos
        # #abs_path_pos = os.path.join(self.base_path, rpath_pos)
        # # read in small chunk
        # speech_arr_pos = sf.SoundFile(abs_path_pos)
        # length_arr_pos = speech_arr_pos.frames
        # if length_arr_pos >= self.window_len:
        #     max_start_pos = length_arr_pos - self.window_len
        #     start_ind_pos = random.randint(0, max_start_pos)
        #     seg_2_bar, _ = sf.read(abs_path_pos, start=start_ind_pos, stop=start_ind_pos+self.window_len)
        # else:
        #     seg_2_bar, _ = sf.read(abs_path_pos)
        # min_snr_dB = 0
        # max_snr_dB = 20    
        
        # 生成一个介于 0 和 1 之间的随机数
        
        probability = random.random()    
        if probability < 0.5:
            if probability < 0.125:
                if len(seg_1) >= 24000:
                    seg_2 = np.concatenate([seg_1_neg, seg_1[len(seg_1_neg):]])
                else:
                    seg_2 = np.concatenate([seg_1_neg, seg_1])
            elif probability >= 0.125 and probability <= 0.25:
                if len(seg_1) >= 24000:
                    seg_2 = np.concatenate([seg_1[:len(seg_1) - len(seg_1_neg)], seg_1_neg])
                else:
                    seg_2 = np.concatenate([seg_1, seg_1_neg])
            else:
                # mean_square1 = np.mean(seg_1 ** 2)
                # mean_square2 = np.mean(seg_1_neg ** 2)
                # # print('mean_square2', mean_square2)
                # rms_db1 = 10 * np.log10(mean_square1)
                # rms_db2 = 10 * np.log10(mean_square2)
                # snr_dB = random.uniform(min_snr_dB, max_snr_dB)                
                
                diff_duration = len(seg_1) - len(seg_1_neg)
                if diff_duration >= 0:
                    # seg_1_neg_new = np.roll(seg_1_neg, shift=diff_duration)[:len(seg_1)]
                    seg_1_neg_new = np.pad(seg_1_neg, (0, diff_duration), 'wrap')
                else:
                    seg_1_neg_new = seg_1_neg[:len(seg_1)]
                # seg_gain_db = min(rms_db1 - rms_db2 - snr_dB, 300)
                # seg_1_neg_new *= 10.**(seg_gain_db / 20.)

                
                seg_2 = seg_1 + seg_1_neg_new
        else:
            seg_2 = seg_1
         
        noverlap = 240
        winlen = 400
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
                winlen, 16000, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        # seg_1 = features.add_dither((seg_1*2**15).astype(int))
        seg_2 = features.add_dither((seg_2*2**15).astype(int))
        # seg_1 = np.r_[seg_1[noverlap // 2 - 1::-1],
        #         seg_1, seg_1[-1:-winlen // 2 - 1:-1]]
        seg_2 = np.r_[seg_2[noverlap // 2 - 1::-1],
                seg_2, seg_2[-1:-winlen // 2 - 1:-1]]
        # fea1 = features.fbank_htk(seg_1, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        # seg_1 = features.cmvn_floating_kaldi(fea1, LC, RC, norm_vars=False).astype(np.float32)
        fea2 = features.fbank_htk(seg_2, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        seg_2 = features.cmvn_floating_kaldi(fea2, LC, RC, norm_vars=False).astype(np.float32)
        
        # seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)
        
        return seg_2, curr_speaker
    
class Cnceleb1ft_sample_cssl(Dataset):
    
    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both CN 12
    """

    def __init__(self, base_path, data_df, seed=0, window_len=32000):
        self.base_path = base_path 
        self.rel_path = data_df['rel_path']
        print(self.rel_path)
        self.speaker = data_df['speaker_id']
        self.len1 = len(data_df)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        self.speaker = list(self.speaker)
        
        #unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            print(sp)
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        # print('index', index)
        
        rpath = self.rel_path[index]
        # abs_path = os.path.join(self.base_path, rpath)
        abs_path = rpath
        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames
        if length_arr >= self.window_len:
            max_start = length_arr - self.window_len
            start_ind = random.randint(0, max_start)
            seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)
        else:
            seg_1, _ = sf.read(abs_path)

        # for hard pos example
        curr_speaker = self.speaker[index]
        neg_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp != curr_speaker])
        rpath_neg = self.rel_path[neg_pair_ind]
        abs_path_neg = rpath_neg
        speech_arr_neg = sf.SoundFile(abs_path_neg)
        length_arr_neg = speech_arr_neg.frames
        if length_arr_neg >= 8000:
            max_start_neg = length_arr_neg - 8000
            start_ind_neg = random.randint(0, max_start_neg)
            duration = random.randint(16, 8000)
            seg_1_neg, _ = sf.read(abs_path_neg, start=start_ind_neg, stop=start_ind_neg+duration)
        else:
            seg_1_neg, _ = sf.read(abs_path_neg)          
        # print('seg_1_neg', seg_1_neg)    
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])

        rpath_pos = self.rel_path[pos_pair_ind]
        abs_path_pos = rpath_pos
        #abs_path_pos = os.path.join(self.base_path, rpath_pos)
        # read in small chunk
        speech_arr_pos = sf.SoundFile(abs_path_pos)
        length_arr_pos = speech_arr_pos.frames
        if length_arr_pos >= self.window_len:
            max_start_pos = length_arr_pos - self.window_len
            start_ind_pos = random.randint(0, max_start_pos)
            seg_2_bar, _ = sf.read(abs_path_pos, start=start_ind_pos, stop=start_ind_pos+self.window_len)
        else:
            seg_2_bar, _ = sf.read(abs_path_pos)
        min_snr_dB = 0
        max_snr_dB = 20    
        # Generate a random number between 0 and 1
        probability = random.random()    
        if probability < 0.5:
            if probability < 0.125:
                seg_2 = np.concatenate([seg_1_neg, seg_1])
            elif probability >= 0.125 and probability <= 0.25:
                seg_2 = np.concatenate([seg_1, seg_1_neg])
            else:
                mean_square1 = np.mean(seg_1 ** 2)
                mean_square2 = np.mean(seg_1_neg ** 2)
                # print('mean_square2', mean_square2)
                rms_db1 = 10 * np.log10(mean_square1)
                rms_db2 = 10 * np.log10(mean_square2)
                snr_dB = random.uniform(min_snr_dB, max_snr_dB)                
                
                diff_duration = len(seg_1) - len(seg_1_neg)
                if diff_duration >= 0:
                    # seg_1_neg_new = np.roll(seg_1_neg, shift=diff_duration)[:len(seg_1)]
                    seg_1_neg_new = np.pad(seg_1_neg, (0, diff_duration), 'wrap')
                else:
                    seg_1_neg_new = seg_1_neg[:len(seg_1)]
                seg_gain_db = min(rms_db1 - rms_db2 - snr_dB, 300)
                seg_1_neg_new *= 10.**(seg_gain_db / 20.)

                
                seg_2 = seg_1 + seg_1_neg_new
        else:
            seg_2 = seg_2_bar
            
        noverlap = 240
        winlen = 400
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
                winlen, 16000, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        seg_1 = features.add_dither((seg_1*2**15).astype(int))
        seg_2 = features.add_dither((seg_2*2**15).astype(int))
        seg_1 = np.r_[seg_1[noverlap // 2 - 1::-1],
                seg_1, seg_1[-1:-winlen // 2 - 1:-1]]
        seg_2 = np.r_[seg_2[noverlap // 2 - 1::-1],
                seg_2, seg_2[-1:-winlen // 2 - 1:-1]]
        fea1 = features.fbank_htk(seg_1, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        seg_1 = features.cmvn_floating_kaldi(fea1, LC, RC, norm_vars=False).astype(np.float32)
        fea2 = features.fbank_htk(seg_2, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        seg_2 = features.cmvn_floating_kaldi(fea2, LC, RC, norm_vars=False).astype(np.float32)
        
        seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)
        
        return seg_1, seg_2, curr_speaker    
    
    
class Cnceleb_sample_cssl(Dataset):
    
    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both CN 12
    """

    def __init__(self, base_path, data_df, seed=0, window_len=46000):
        self.base_path = base_path
        
        self.rel_path = data_df['rel_path']
        # print(self.rel_path)
        self.speaker = data_df['speaker_id']
        
        self.len1 = len(data_df)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        self.speaker = list(self.speaker)
        
        #unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            # print(sp)
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        # print('index', index)
        
        
        # start_time = time.time()
        rpath = self.rel_path[index]
        # abs_path = os.path.join(self.base_path, rpath)
        abs_path = rpath
        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames
        if length_arr >= self.window_len:
            max_start = length_arr - self.window_len
            start_ind = random.randint(0, max_start)
            seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)
        else:
            seg_1, _ = sf.read(abs_path)

        # for hard pos example
        curr_speaker = self.speaker[index]
        neg_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp != curr_speaker])
        rpath_neg = self.rel_path[neg_pair_ind]
        abs_path_neg = rpath_neg
        speech_arr_neg = sf.SoundFile(abs_path_neg)
        # print('abs_path_neg', abs_path_neg)
        length_arr_neg = speech_arr_neg.frames
        # print('length_arr_neg', length_arr_neg)

        if length_arr_neg >= 10000:
            max_start_neg = length_arr_neg - 9000
            start_ind_neg = random.randint(0, max_start_neg)
            duration = random.randint(160, 8000)
            # print('duration, start_ind_neg', duration, start_ind_neg)
            seg_1_neg, _ = sf.read(abs_path_neg, start=start_ind_neg, stop=start_ind_neg+duration)
        else:
            seg_1_neg, _ = sf.read(abs_path_neg)  
       
        # print('length_arr_neg', len(seg_1_neg))        
        # print('seg_1_neg', seg_1_neg)    
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        # pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])

        # rpath_pos = self.rel_path[pos_pair_ind]
        # abs_path_pos = rpath_pos
        # #abs_path_pos = os.path.join(self.base_path, rpath_pos)
        # # read in small chunk
        # speech_arr_pos = sf.SoundFile(abs_path_pos)
        # length_arr_pos = speech_arr_pos.frames
        # if length_arr_pos >= self.window_len:
        #     max_start_pos = length_arr_pos - self.window_len
        #     start_ind_pos = random.randint(0, max_start_pos)
        #     seg_2_bar, _ = sf.read(abs_path_pos, start=start_ind_pos, stop=start_ind_pos+self.window_len)
        # else:
        #     seg_2_bar, _ = sf.read(abs_path_pos)
        # min_snr_dB = 0
        # max_snr_dB = 20    
        
        # Generate a random number between 0 and 1
        
        probability = random.random()    
        if probability < 0.5:
            if probability < 0.125:
                if len(seg_1) >= 24000:
                    seg_2 = np.concatenate([seg_1_neg, seg_1[len(seg_1_neg):]])
                else:
                    seg_2 = np.concatenate([seg_1_neg, seg_1])
            elif probability >= 0.125 and probability <= 0.25:
                if len(seg_1) >= 24000:
                    seg_2 = np.concatenate([seg_1[:len(seg_1) - len(seg_1_neg)], seg_1_neg])
                else:
                    seg_2 = np.concatenate([seg_1, seg_1_neg])
            else:
                # mean_square1 = np.mean(seg_1 ** 2)
                # mean_square2 = np.mean(seg_1_neg ** 2)
                # # print('mean_square2', mean_square2)
                # rms_db1 = 10 * np.log10(mean_square1)
                # rms_db2 = 10 * np.log10(mean_square2)
                # snr_dB = random.uniform(min_snr_dB, max_snr_dB)                
                
                diff_duration = len(seg_1) - len(seg_1_neg)
                if diff_duration >= 0:
                    # seg_1_neg_new = np.roll(seg_1_neg, shift=diff_duration)[:len(seg_1)]
                    seg_1_neg_new = np.pad(seg_1_neg, (0, diff_duration), 'wrap')
                else:
                    seg_1_neg_new = seg_1_neg[:len(seg_1)]
                # seg_gain_db = min(rms_db1 - rms_db2 - snr_dB, 300)
                # seg_1_neg_new *= 10.**(seg_gain_db / 20.)

                
                seg_2 = seg_1 + seg_1_neg_new
        else:
            seg_2 = seg_1

        noverlap = 240
        winlen = 400
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
                winlen, 16000, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        # seg_1 = features.add_dither((seg_1*2**15).astype(int))
        seg_2 = features.add_dither((seg_2*2**15).astype(int))
        # seg_1 = np.r_[seg_1[noverlap // 2 - 1::-1],
        #         seg_1, seg_1[-1:-winlen // 2 - 1:-1]]
        seg_2 = np.r_[seg_2[noverlap // 2 - 1::-1],
                seg_2, seg_2[-1:-winlen // 2 - 1:-1]]
        # fea1 = features.fbank_htk(seg_1, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        # seg_1 = features.cmvn_floating_kaldi(fea1, LC, RC, norm_vars=False).astype(np.float32)
        fea2 = features.fbank_htk(seg_2, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        seg_2 = features.cmvn_floating_kaldi(fea2, LC, RC, norm_vars=False).astype(np.float32)
        
        # seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)
        
        return seg_2, curr_speaker


class Cnceleb_sample_cssl2(Dataset):
    
    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both CN 12
    """

    def __init__(self, base_path, data_df, seed=0, window_len=48000):
        self.base_path = base_path
        
        self.rel_path = data_df['rel_path']
        # print(self.rel_path)
        self.speaker = data_df['speaker_id']
        
        self.len1 = len(data_df)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        self.speaker = list(self.speaker)
        
        #unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            # print(sp)
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        # print('index', index)
        
        
        # start_time = time.time()
        rpath = self.rel_path[index]
        # abs_path = os.path.join(self.base_path, rpath)
        abs_path = rpath
        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames
        if length_arr >= self.window_len:
            max_start = length_arr - self.window_len
            start_ind = random.randint(0, max_start)
            seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)
        else:
            seg_1, _ = sf.read(abs_path)
        
        curr_speaker = self.speaker[index]
        # print('len1,len2', len(seg_1), len(seg_2))    
        noverlap = 240
        winlen = 400
        window = features.povey_window(winlen)
        fbank_mx = features.mel_fbank_mx(
                winlen, 16000, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        LC = 150
        RC = 149
        # seg_1 = features.add_dither((seg_1*2**15).astype(int))
        seg_1 = features.add_dither((seg_1*2**15).astype(int))
        # seg_1 = np.r_[seg_1[noverlap // 2 - 1::-1],
        #         seg_1, seg_1[-1:-winlen // 2 - 1:-1]]
        seg_1 = np.r_[seg_1[noverlap // 2 - 1::-1],
                seg_1, seg_1[-1:-winlen // 2 - 1:-1]]
        # fea1 = features.fbank_htk(seg_1, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        # seg_1 = features.cmvn_floating_kaldi(fea1, LC, RC, norm_vars=False).astype(np.float32)
        fea1 = features.fbank_htk(seg_1, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        seg_1 = features.cmvn_floating_kaldi(fea1, LC, RC, norm_vars=False).astype(np.float32)
        
        # seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        
        return seg_1, curr_speaker
    
    

class Wav2VecSpeakerCSSL(torch.nn.Module):

    def __init__(self, w2v2_model, config):
        super().__init__()
        self.config = config
        self.config_w2v2 = self.config['w2v2_config']

        self.model = w2v2_model   
        self.dropout = torch.nn.Dropout(self.config['dropout_val'] if 'dropout_val' in self.config else 0.2)
        self.with_relu = config['with_relu']
        self.relu = torch.nn.ReLU()
        self.layer_to_extract = config['layer_to_extract'] if 'layer_to_extract' in self.config else -1


        self.loss_fn = AngProtoLoss4(
            config,
            device=self.config["device"], 
            refine_matrix=config['refine_matrix'], 
            g_blur=config['g_blur'],
            p_pct=config['p_pct'],
            mse_fac=config['mse_fac'])

        if self.config['custom_embed_size']:
            self.fc1 = torch.nn.Linear(768, self.config["custom_embed_size"]) 
            self.init_weights()

    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, seg_1, seg_2, labels=None):
        f1 = self.model(seg_1).last_hidden_state
        f2 = self.model(seg_2).last_hidden_state

        f1 = torch.mean(f1, dim=1)
        f2 = torch.mean(f2, dim=1)

        f_all = torch.stack((f1, f2), dim=1)
        if self.with_relu:
            f_all = self.relu(f_all)

        if self.config['custom_embed_size']:
            f_all = self.fc1(f_all)
            if self.with_relu:
                f_all = self.relu(f_all)

        loss = self.loss_fn(f_all, labels)

        return loss, f_all

    def extract_embeddings(self, input_values):
        
        features = self.model(input_values, output_hidden_states=True)
        features = features.hidden_states[self.layer_to_extract]
        pooled_output = torch.mean(features, dim=1)
        if self.with_relu:
            pooled_output = self.relu(pooled_output)

        if self.config['custom_embed_size']:
            pooled_output = self.fc1(pooled_output)
            if self.with_relu:
                pooled_output = self.relu(pooled_output)

        return pooled_output


class Resnet101SpeakerCSSL(torch.nn.Module):
    
    def __init__(self, resnet_model, config):
        super().__init__()
        self.config = config
        # self.config_w2v2 = self.config['w2v2_config']
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = resnet_model   
        # self.dropout = torch.nn.Dropout(self.config['dropout_val'] if 'dropout_val' in self.config else 0.2)
        # self.with_relu = config['with_relu']
        # self.relu = torch.nn.ReLU()
        # self.layer_to_extract = config['layer_to_extract'] if 'layer_to_extract' in self.config else -1
        if self.config['custom_embed_size']:
            self.fc1 = torch.nn.Linear(256, self.config["custom_embed_size"]) 
            self.init_weights()

        self.loss_fn = AngProtoLoss4(
            config,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), 
            refine_matrix=config['refine_matrix'], 
            g_blur=config['g_blur'],
            p_pct=config['p_pct'],
            mse_fac=config['mse_fac'])


    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, seg_2, labels=None):
        seg_2 = seg_2.to(self.device)
        labels = labels.to(self.device)

        seg_2 = torch.transpose(seg_2, 1, 2)
        f2 = self.extract_embeddings(seg_2)
        f2 = f2.data.cpu().numpy()

        loss = self.loss_fn(f2, labels)

        return loss, f2

    def extract_embeddings(self, input_values):
        
        embeddings = self.model(input_values)

        return embeddings
