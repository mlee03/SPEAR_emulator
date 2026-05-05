from types import SimpleNamespace

import lightning as pl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import xarray as xr

from base import (
    TrainingDataset,
    TestingDataset,
    SimpleCNN
)


class AutoTrainModule(pl.LightningModule):
    """A Lightning auto training model."""

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)        

        fig, ax = plt.subplots()
        ax.plot([target.detach().to("cpu").mean() for target in targets], label="target mean")
        ax.plot([iz.detach().to("cpu").mean() for iz in z], label="prediction mean")
        ax.legend()

        self.logger.experiment.add_figure("prediction vs target", fig, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class AutoDataModule(pl.LightningDataModule):
    def __init__(self, datafile, variable, lag: int = 3, testsize = 0.2, valsize = 0.3):
        super().__init__()
        self.save_hyperparameters()

        self.datafile = datafile
        self.variable = variable
        self.lag = lag

        self.ds = SimpleNamespace(
            training = None,
            val = None, 
            testing = None
        )

        self.testsize = testsize
        self.valsize = valsize


    def setup(self, stage = None, training = True):
        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            data = ds[self.variable].values
            norm = data.mean()
            data = data/norm
            ntimes = ds.sizes["time"]
            time = list(range(ntimes))

        train_time, testing_time, train_ds, testing_ds = train_test_split(
            time, data, test_size=self.testsize, shuffle=False)

        train_time, val_time, training_ds, val_ds = train_test_split(
            train_time, train_ds, test_size=self.valsize, shuffle=False)

        if training:
            self.ds.training = TrainingDataset(training_ds, lag=self.lag)
            self.ds.val = TrainingDataset(val_ds, lag=self.lag)
        else:
            self.ds.training = TestingDataset(training_ds, lag=self.lag, time=train_time)
            self.ds.val = TestingDataset(val_ds, lag=self.lag, time=val_time)

    def train_dataloader(self, batch_size: int = 32):
        """
        Returns a DataLoader for the training dataset.
        """
        return torch.utils.data.DataLoader(
            self.ds.training, batch_size=batch_size, shuffle=False
        )

    def val_dataloader(self, batch_size: int = 32):
        """
        Returns a DataLoader for the validation dataset.
        """
        return torch.utils.data.DataLoader(
            self.ds.val, batch_size=batch_size, shuffle=False
        )