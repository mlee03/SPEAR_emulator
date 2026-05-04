import lightning as pl
from sklearn.model_selection import train_test_split
import torch
import xarray as xr

from base import (
    AutoregressiveDataset,
    SimpleCNN
)


class LitAutoCNN(pl.LightningModule):
    """A Lightning autoencoder model."""

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.mse_loss(z, y)
        self.log("train_loss", loss)
        self.log("mean temperature", z.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.mse_loss(z, y)
        self.log("val_loss", loss)
        self.log("mean temperature", z.mean())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.mse_loss(z, y)
        self.log("test_loss", loss)
        self.log("mean temperature", z.mean())
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

class AutoDataModule(pl.LightningDataModule):
    def __init__(self, datafile, variable, lag: int = 3, testsize=0.2, valsize=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.datafile = datafile
        self.variable = variable
        self.lag = lag
        self.testsize = testsize
        self.valsize = valsize

        self.training_ds = None
        self.testing_ds = None
        self.val_ds = None

    def setup(self, stage = None):
        with xr.open_dataset(self.datafile, decode_timedelta=True) as ds:
            data = ds[self.variable].values
            norm = data.mean()
            data = data/norm

        train_val, self.testing_ds = train_test_split(
            data, test_size=self.testsize, shuffle=False)[:2]

        self.training_ds, self.val_ds = train_test_split(
            train_val, test_size=self.valsize, shuffle=False)[:2]

        self.training_ds = AutoregressiveDataset(self.training_ds, lag=self.lag)
        self.testing_ds = AutoregressiveDataset(self.testing_ds, lag=self.lag)
        self.val_ds = AutoregressiveDataset(self.val_ds, lag=self.lag)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_ds, batch_size=32, shuffle=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=32, shuffle=False
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testing_ds, batch_size=32, shuffle=False
        )
