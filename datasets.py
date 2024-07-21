import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data, ecg_tracings, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data, ecg_tracings, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations, batch_size, end_idx=n_train)
        valid_seq = cls(C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data, ecg_tracings, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data, ecg_tracings, C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations is None:
            self.y = None
        else:
            self.y = pd.read_csv(C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data\annotations).values
        # Get tracings
        self.f = h5py.File(C:\Users\Roshan\Desktop\ECGdetection\automatic-ecg-diagnosis\data, "r")
        self.x = self.f[ecg_tracings]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
