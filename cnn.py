from pathlib import Path

import lightning as pl
from matplotlib import pyplot as plt
import torch

from base import SimpleCNN, PredictAutoregressiveDataset
from auto import AutoregressiveTrainModule, AutoDataModule


cnn = SimpleCNN()
sequence_length = 3

reload = True
train = False

max_epochs = 2000
learning_rate = 1e-3
saved_chkpt_path = Path("lightning_logs/channels-3-6-1/checkpoints/epoch=4999-step=50000.ckpt")

if reload:
    model = AutoregressiveTrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = AutoregressiveTrainModule(cnn, learning_rate=learning_rate)

trainer = pl.Trainer(max_epochs=max_epochs)

if train:
    data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length).setup_simple_autoregressive()
    trainer.fit(model=model, datamodule=data)


# evaluate
model.model.eval()
model.model.cpu()


#first evaluation
data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length, testsize=0.0, valsize=0.0).setup_simple_autoregressive()
for (inputs, targets) in data.train_dataloader(batch_size=len(data.training_ds)):
    with torch.no_grad():
        z = model.model(inputs)

fitted_mean = [iz.detach().mean() for iz in z]
targets_mean = [target.detach().mean() for target in targets]
        
fig1, ax = plt.subplots()
ax.plot(targets_mean, color='black', label='actual')
ax.plot(fitted_mean, color='pink', label='predicted')
ax.legend()


#second evaluation
data = PredictAutoregressiveDataset(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length)
with torch.no_grad():
    for itime in range(sequence_length, data.ntimes):
        inputs = data.get_inputs()
        z = model.model(inputs)
        data.add(z)
        
predicted_mean = [ipredicted.detach().mean() for ipredicted in data.predictions]
answer_mean = [idata.mean() for idata in data.data]

fig2, ax = plt.subplots()
ax.plot(data.time, answer_mean, label='actual')
ax.plot(data.time, predicted_mean, label='predicted')
ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('value')
ax.set_ylim(0.95, 1.05)
plt.show()

if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
    trainer.logger.experiment.add_figure("fits", fig1, global_step=trainer.global_step)
    trainer.logger.experiment.add_figure("predictions", fig2, global_step=trainer.global_step)
    trainer.logger.experiment.flush()


