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
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        for example sequence_length=3, idx=3, returns 
        inputs = [data[3], data[4], data[5]]
        target = data[6] 
        """
        return self.data[idx:idx+self.sequence_length], self.data[idx+self.sequence_length]


class LSTMTrainingDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for LSTM training on 1D time series (e.g. global means).

    Returns:
        inputs: shape (sequence_length, 1) — sequence of scalar values as LSTM features
        target: scalar float — next timestep value
    """

    def __init__(
        self,
        data: np.ndarray = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        for example, sequence_length=3, idx=3, returns
        inputs = [[data[3]], [data[4]], [data[5]]]  # shape (sequence_length, 1)
        target = data[6]  # scalar
        """
        return self.data[idx:idx+self.sequence_length].unsqueeze(-1), self.data[idx+self.sequence_length]

class PredictAutoregressiveDataset():
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        datafile: str|Path = None,
        variable: str = None,
        sequence_length: int = 3,
        normalize: bool = True
    ):
        super().__init__()

        self.datafile = datafile
        self.variable = variable
        self.sequence_length = sequence_length
        self.normalize = normalize

        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            self.data = ds[self.variable].values
            if self.normalize:
                self.norm = self.data.mean()
                self.data = self.data/self.norm
            self.ntimes = len(self.data)
            self.time = list(range(self.ntimes))
        
        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length])

    def get_inputs(self):
        """Returns the last sequence_length predictions as input."""
        return self.predictions[-self.sequence_length:]

    def add(self, value):
        """Adds a new prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.unsqueeze(0)), dim=0)


class PredictLSTMDataset():
    """A dataset for autoregressive prediction with an LSTM model on 1D time series (e.g. global means)."""

    def __init__(
        self,
        datafile: str|Path = None,
        variable: str = None,
        sequence_length: int = 3,
        normalize: bool = True
    ):
        super().__init__()

        self.datafile = datafile
        self.variable = variable
        self.normalize = normalize
        self.sequence_length = sequence_length        

        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            spatial_dims = [d for d in ds[self.variable].dims if d != 'time']
            self.data = ds[self.variable].mean(dim=spatial_dims).values
            if self.normalize:
                self.norm = self.data.mean()
                self.data = self.data / self.norm
            
            self.ntimes = len(self.data)
            self.time = list(range(self.ntimes))               

        # initial predictions seeded from the first sequence_length timesteps
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input of shape (1, sequence_length, 1)."""
        return self.predictions[-self.sequence_length:].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

    def add(self, value):
        """Adds a new scalar prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.detach().reshape(1)), dim=0)


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
        return y.squeeze()

class SimpleLSTM(torch.nn.Module):
    """A simple LSTM model."""

    def __init__(self, input_size: int = 1, hidden_size: int = 10, num_layers: int = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return output.squeeze()