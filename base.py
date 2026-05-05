import numpy as np
from pathlib import Path

import lightning as L
import torch
import xarray as xr


class TrainingDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray = None,
        lag: int = 3,
    ):
        super().__init__()
        self.data = data
        self.lag = lag

    def __len__(self):
        return len(self.data) - self.lag

    def __getitem__(self, idx):
        return self.data[idx:idx+self.lag], self.data[idx+self.lag]


class TestingDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray = None,
        time: list|np.ndarray = None,
        lag: int = 3,
    ):
        super().__init__()
        self.data = data
        self.time = time
        self.lag = lag

    def __len__(self):
        return len(self.data) - self.lag

    def __getitem__(self, idx):
        return self.data[idx:idx+self.lag]


class SimpleCNN(torch.nn.Module):
    """A simple CNN model with one convolutional layer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=2*in_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = torch.nn.ReLU()
        self.cnn2 = torch.nn.Conv2d(
            in_channels=2*in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        y = self.relu(self.cnn1(x))
        y = self.cnn2(y)
        if self.eval():
            return y[:,0,:,:]
        else:
            return y
