import lightning as pl
from base import SimpleCNN
from auto import LitAutoCNN, AutoDataModule
from pathlib import Path


cnn = SimpleCNN()
model = LitAutoCNN(cnn)

data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref")
data.setup()

trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model=model, datamodule=data)

#latest_ckpt = "/home/Mikyung.Lee/spear-emulator-me/lightning_logs/version_0/checkpoints/epoch=999-step=11000.ckpt"
#model = LitAutoCNN.load_from_checkpoint(latest_ckpt, weights_only=False)

model.eval()
trainer.test(model=model, dataloaders=data.test_dataloader())