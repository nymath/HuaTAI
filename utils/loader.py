import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader


class TsDataSet(Dataset):

    offset: int 
    X_2D: typing.Union[np.ndarray, torch.Tensor]
    y: typing.Union[np.ndarray, torch.Tensor]

    def __init__(
            self, 
            X: typing.Union[np.ndarray, torch.Tensor], 
            y: typing.Union[np.ndarray, torch.Tensor], 
            len_seq: int=5, 
            zero_padding=True,
    ):
        self.offset = len_seq
        self.X_2D = X
        self.y = y
        self.X_2D_padded = self._zero_padding()
        self.X_3D = np.concatenate([self.X_2D_padded[index: index+self.offset][np.newaxis, ...] for index in range(len(self.X_2D))], axis=0)

    def _zero_padding(self):
        padding_array = np.zeros(shape=(self.offset-1, self.X_2D.shape[1]), dtype=self.X_2D.dtype)
        X_pad = np.concatenate([padding_array, self.X_2D], axis=0)
        return X_pad    

    def __len__(self):
        return len(self.X_2D)
    
    def __getitem__(self, index) -> typing.Tuple[typing.Union[np.ndarray, torch.Tensor], typing.Union[np.ndarray, torch.Tensor]]:
        return self.X_3D[index], self.y[index]