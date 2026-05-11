from pathlib import Path

import lightning as pl
from matplotlib import pyplot as plt
import torch

from base import SimpleLSTM, PredictLSTMDataset
from auto import AutoregressiveTrainModule, AutoDataModule

input_size = 1
sequence_length = 5
lstm = SimpleLSTM(input_size=input_size)


reload = True
train = True

max_epochs = 1000
learning_rate = 1e-5
saved_chkpt_path = Path("lightning_logs/version_0/checkpoints/epoch=4999-step=50000.ckpt")

trainer = pl.Trainer(max_epochs=max_epochs)
if reload:
    model = AutoregressiveTrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = AutoregressiveTrainModule(lstm, learning_rate=learning_rate)

if train:
    data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length).setup_simple_lstm()
    trainer.fit(model=model, datamodule=data)


# evaluate
model.model.eval()
model.model.cpu()


#first evaluation
data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length, trainingsize=0.999, valsize=0.001).setup_simple_lstm()
for (inputs, targets) in data.train_dataloader(batch_size=len(data.training_ds)):
    with torch.no_grad():
        z = model.model(inputs)
        
fig1, ax = plt.subplots()
ax.plot(targets.detach(), color='black', label='actual')
ax.plot(z.detach(), color='pink', label='fitted')
ax.legend()


#second evaluation
data = PredictLSTMDataset(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", sequence_length=sequence_length)
with torch.no_grad():
    for itime in range(sequence_length, data.ntimes):
        inputs = data.get_inputs()
        z = model.model(inputs)        
        data.add(z)
        
fig2, ax = plt.subplots()
ax.plot(data.time, data.data, label='actual')
ax.plot(data.time, data.predictions.detach().numpy(), label='predicted')
ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('value')
plt.show()

if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
    trainer.logger.experiment.add_figure("fits", fig1, global_step=trainer.global_step)
    trainer.logger.experiment.add_figure("predictions", fig2, global_step=trainer.global_step)
    trainer.logger.experiment.flush()


