import lightning as pl
from base import SimpleCNN
from auto import AutoTrainModule, AutoDataModule
from pathlib import Path


cnn = SimpleCNN()

reload = False
train = False

if reload:
    latest_ckpt = "/home/Mikyung.Lee/spear-emulator-me/lightning_logs/version_0/checkpoints/epoch=999-step=11000.ckpt"
    model = AutoTrainModule.load_from_checkpoint(latest_ckpt, weights_only=False)
else:
    model = AutoTrainModule(cnn)

data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref")
data.setup()

trainer = pl.Trainer(max_epochs=1)

if train:
    trainer.fit(model=model, datamodule=data)

model.eval()
y = trainer.predict(model=model, dataloaders=data.val_dataloader())
print(y)