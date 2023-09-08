# ----------------------------------------------------------------------------------------------------------------------
# /*==========================================================*\
# |     ||  -          Copyright Piovesan            - ||      |
# |     ||  -       Audio Features Extraction        - ||      |
# |     ||  -          By: Thiago Piovesan           - ||      |
# |     ||  -          Versao atual: 1.0.0           - ||      |
# \*==========================================================*/
#   This software is confidential and property of NoLeak.
#   Your distribution or disclosure of your content is not permitted without express permission from NoLeak.
#   This file contains proprietary information.

#   Link do Github: https://github.com/ThiagoPiovesan
# ----------------------------------------------------------------------------------------------------------------------
# Libs Importation:
import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset 

ANNOTATIONS_FILE: str = "C:/Users/thiag/Datasets/UrbanSound8K/metadata/UrbanSound8k.csv"
AUDIO_DIR: str = "C:/Users/thiag/Datasets/UrbanSound8K/audio"
SAMPLE_RATE: int = 22050
NUM_SAMPLES: int = 22050

class UrbanSoundDataset(Dataset):
    
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device) -> None:
        self.device = device
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate
        self.annotations = pd.read_csv(annotations_file)
        self.transformation = transformation.to(self.device)
        
    def __len__(self):
        # len(usd) -> How to calculate this, number of samples
        return len(self.annotations)
    
    def __getitem__(self, index):
        # a_list[1] -> a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)    
        
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        signal = self.transformation(signal)
        
        return signal, label
    
    def _cut_if_necessary(self, signal):
        """
            When we have more samples than expected -> cut the signal
        """
        
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
            
        return signal
        
    def _right_pad_if_necessary(self, signal):
        """
            When we have less samples than expected -> add the signal to the right
        """
        length_signal = signal.shape[1]

        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length_signal
            
            last_dim_padding = (0, num_missing_samples) # (0, 2) / (1, 2) -> (left, right)
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            # [1, 1, 1] -> [0, 1, 1, 1, 0, 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        """
            Not all the signals on dataset have the same sample rate, so you need to normalize (padronizar)
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            
        return signal
    
    def _mix_down_if_necessary(self, signal):
        """
            Not all the signals have just one channel (Mono), some are Stereo (two channels) or more, so we need to normalize this
        """ 
        
        if signal.shape[0] > 1: # (2, 1000) -> The first parameter is the number of channels
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        return signal   

    def _get_audio_sample_path(self, index) -> str:
        """
            Get the index and return the path to access the audio sample.
        """
        
        fold = f"fold{self.annotations.iloc[index, 5]}"    # fold5, fold8, ...
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        
        return path.replace("\\", "/")
    
    def _get_audio_sample_label(self, index) -> str:
        """
            Get the index and return the label to the respective audio sample.
        """
        return self.annotations.iloc[index, 6]
        
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Using device: {}".format(device))
    
    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    print("There are {} samples in the dataset".format(len(usd)))
    signal, label = usd[0]