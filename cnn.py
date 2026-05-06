from pathlib import Path

import lightning as pl
from matplotlib import pyplot as plt

from base import SimpleCNN, PredictDataset
from auto import AutoTrainModule, AutoDataModule



cnn = SimpleCNN()

reload = True
train = False

if reload:
    latest_ckpt = "/home/Mikyung.Lee/spear-emulator-me/lightning_logs/version_0/checkpoints/epoch=4999-step=50000.ckpt"
    model = AutoTrainModule.load_from_checkpoint(latest_ckpt, weights_only=False, map_location="cpu")
else:
    model = AutoTrainModule(cnn)


if train:
    data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref")
    data.setup()
    trainer = pl.Trainer(max_epochs=5000)
    trainer.fit(model=model, datamodule=data)

# evaluate
cnn.eval()
cnn.cpu()
data = PredictDataset(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", lag=3)

for itime in range(data.ntimes):
    print(itime)    
    inputs = data.get_inputs()
    z = cnn(inputs)    
    data.add(z)

predicted_mean = [ipredicted.detach().mean() for ipredicted in data.predictions]

fig, ax = plt.subplots()
ax.plot(predicted_mean)
plt.show()


