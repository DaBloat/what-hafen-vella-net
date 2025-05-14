import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio.functional as F
import pandas as pd


class MFCCData(Dataset):
    """This Dataset Set up is used for the BI-LTSM Model"""
    def __init__(self, data, aud, label, max_len=300):
        self.data = pd.read_csv(data)
        self.audio_files = aud
        self.label_map = label
        self.sample_rate = 16000
        self.max_len = max_len
        
        self.to_mfcc = T.MFCC(
            n_mfcc = 40,
            sample_rate = self.sample_rate,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 64,
                "center": True,
                "power": 2.0
            }
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        wav = os.path.join(self.audio_files, f'dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}' + '.wav')
        label = self.label_map[row['Emotion']]
        
        # Extract MFCCs and format [T, 40]
        mfcc = self.to_mfcc(wav).squeeze(0).transpose(0, 1)

        # Pad or truncate
        T_current, D = mfcc.shape
        if T_current < self.max_len:
            pad = torch.zeros(self.max_len - T_current, D)
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:self.max_len, :]
            
        return mfcc, torch.tensor(label)