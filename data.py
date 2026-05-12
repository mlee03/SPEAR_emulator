import numpy as np
from pathlib import Path

import lightning as pl
import torch
import xarray as xr
from sklearn.model_selection import train_test_split


class TrainingDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NetCDF data."""

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


def load_variable(datafile: str | Path, variable: str) -> tuple[np.ndarray, int, list[int]]:
    """Open a NetCDF file and return (values, ntimes, time) for the given variable."""
    with xr.open_dataset(datafile, decode_timedelta=True) as ds:
        data = ds[variable].values
        ntimes = len(data)
        time = list(range(ntimes))
    return data, ntimes, time


class _BaseDataModule(pl.LightningDataModule):
    """Shared logic for autoregressive data modules."""

    def __init__(self, sequence_length: int = 3, trainingsize: float = 0.8, valsize: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.trainingsize = trainingsize
        self.valsize = valsize
        self.norm = None
        self.training_ds = None
        self.val_ds = None

    def _split(self, data):
        train, val = train_test_split(
            train_test_split(data, train_size=self.trainingsize, shuffle=False)[0],
            test_size=self.valsize,
            shuffle=False,
        )
        return train, val

    def _print_split_sizes(self, train, val):
        print(f"Training size: {len(train)}")
        print(f"Validation size: {len(val)}")

    def train_dataloader(self, batch_size: int = 32):
        """Load training data onto DataLoader"""
        print(f"Training dataset size: {len(self.training_ds)}")
        return torch.utils.data.DataLoader(self.training_ds, batch_size=batch_size, shuffle=False)

    def val_dataloader(self, batch_size: int = 32):
        """Load validation data onto DataLoader"""
        print(f"Validation dataset size: {len(self.val_ds)}")
        return torch.utils.data.DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)


class AutoregressiveDataModule(_BaseDataModule):
    """Data module for spatial autoregressive (CNN-style) training."""

    def __init__(self, data: np.ndarray, sequence_length: int = 3, trainingsize: float = 0.8, valsize: float = 0.3):
        super().__init__(sequence_length=sequence_length, trainingsize=trainingsize, valsize=valsize)
        self._raw = data

    def setup(self, stage = None, normalize: bool = True):
        """setup data"""
        data = np.array(self._raw)
        if normalize:
            self.norm = np.mean(data)
            data = data / self.norm

        train, val = self._split(data)
        self._print_split_sizes(train, val)
        self.training_ds = TrainingDataset(train, sequence_length=self.sequence_length)
        self.val_ds = TrainingDataset(val, sequence_length=self.sequence_length)

        return self


class AutoLSTMDataModule(_BaseDataModule):
    """Data module for LSTM training on 1D global-mean time series."""

    def __init__(self, data: np.ndarray, sequence_length: int = 3, trainingsize: float = 0.8, valsize: float = 0.3):
        super().__init__(sequence_length=sequence_length, trainingsize=trainingsize, valsize=valsize)
        self._raw = data

    def setup(self, stage = None, normalize: bool = True):
        """setup data"""
        data = np.array([datum.mean() for datum in self._raw])
        if normalize:
            self.norm = data.mean()
            data = data / self.norm

        train, val = self._split(data)
        self._print_split_sizes(train, val)
        self.training_ds = LSTMTrainingDataset(train, sequence_length=self.sequence_length)
        self.val_ds = LSTMTrainingDataset(val, sequence_length=self.sequence_length)

        return self


class PredictAutoregressiveDataset():
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 3,
        normalize: bool = True
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.normalize = normalize

        self.data = np.array(data)
        self.ntimes = len(self.data)
        self.time = list(range(self.ntimes))
        if self.normalize:
            self.norm = self.data.mean()
            self.data = self.data / self.norm

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

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
        data: np.ndarray,
        sequence_length: int = 3,
        normalize: bool = True
    ):
        super().__init__()

        self.normalize = normalize
        self.sequence_length = sequence_length

        self.data = np.array([datum.mean() for datum in data])
        self.ntimes = len(self.data)
        self.time = list(range(self.ntimes))
        if self.normalize:
            self.norm = np.mean(self.data)
            self.data = self.data / self.norm

        # initial predictions seeded from the first sequence_length timesteps
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input of shape (1, sequence_length, 1)."""
        return self.predictions[-self.sequence_length:].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

    def add(self, value):
        """Adds a new scalar prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.detach().reshape(1)), dim=0)


