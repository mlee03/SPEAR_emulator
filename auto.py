from types import SimpleNamespace

import lightning as pl
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import xarray as xr

from base import TrainingDataset, LSTMTrainingDataset


class AutoregressiveTrainModule(pl.LightningModule):
    """A Lightning auto training model."""

    def __init__(self, model, learning_rate=1e-3):
        """Constructor"""
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        """The training step"""
        inputs, targets = batch
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        inputs, targets = batch
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        """ Configure Adam optimizer for training. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class AutoDataModule(pl.LightningDataModule):
    """ Data handler """
    def __init__(self, datafile, variable, sequence_length: int = 3, trainingsize = 0.8, valsize = 0.3):
        super().__init__()
        self.save_hyperparameters()

        self.datafile = datafile
        self.variable = variable
        self.sequence_length = sequence_length
        self.norm = None

        self.training_ds = None
        self.val_ds = None        

        self.trainingsize = trainingsize
        self.valsize = valsize

    def setup_simple_autoregressive(self, stage = None, normalize: bool = True):
        """ set self.ds """
        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            data = np.array(ds[self.variable].values)
            if normalize:
                self.norm = data.mean()
                data = data/self.norm                

        # split training to training and validatin
        training_ds, val_ds = train_test_split(
            train_test_split(data, train_size=self.trainingsize, shuffle=False)[0],
            test_size=self.valsize, 
            shuffle=False
        )

        self.training_ds = TrainingDataset(training_ds, sequence_length=self.sequence_length)
        self.val_ds = TrainingDataset(val_ds, sequence_length=self.sequence_length)
        
        return self

    def setup_simple_lstm(self, stage = None, normalize: bool = True):
        """ set self.ds """
        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            data = np.array([idata.mean() for idata in ds[self.variable].values])
            if normalize:
                self.norm = data.mean()
                data = data/self.norm

        # split training to training and validatin
        training_ds, val_ds = train_test_split(
            train_test_split(data, train_size=self.trainingsize, shuffle=False)[0],
            test_size=self.valsize,
            shuffle=False
        )
        
        self.training_ds = LSTMTrainingDataset(training_ds, sequence_length=self.sequence_length)
        self.val_ds = LSTMTrainingDataset(val_ds, sequence_length=self.sequence_length)
        
        return self

    def train_dataloader(self, batch_size: int = 32):
        """
        Returns a DataLoader for the training dataset.
        """
        return torch.utils.data.DataLoader(
            self.training_ds, batch_size=batch_size, shuffle=False
        )

    def val_dataloader(self, batch_size: int = 32):
        """
        Returns a DataLoader for the validation dataset.
        """
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False
        )