import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import numpy as np
import torch

from model import TrainModule, SimpleLSTM
from data import (
    AutoDataModule, 
    PredictLSTMDataset, 
    TrainingLSTMDataset, 
    load_variable, 
    normalize
)

#read data
tref = load_variable("data/atmos.192101-201012.t_ref.nc", "t_ref")[0]
tref_mean = normalize([datum.mean() for datum in tref])

swdn_toa = load_variable("data/atmos.192101-201012.swdn_toa.nc", "swdn_toa")[0]
swdn_toa_mean = normalize([datum.mean() for datum in swdn_toa])
labels = ['tref', 'swdn_toa']

inputs = np.column_stack((tref_mean, swdn_toa_mean))

#lstm parameters
input_size = 2
sequence_length = 5
lstm = SimpleLSTM(input_size=input_size, output_size=2)

reload = True
train = True

max_epochs = 5
learning_rate = 1e-3
saved_chkpt_path = "/home/Mikyung.Lee/spear-emulator-me/lstm-2variable/version_3/checkpoints/epoch=1999-step=22000.ckpt"


if reload:
    if saved_chkpt_path is None:
        raise ValueError("Set saved_chkpt_path before using reload=True")
    model = TrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = TrainModule(lstm, learning_rate=learning_rate)

if train:
    tb_logger = TensorBoardLogger(save_dir="lstm-2variable-2outputs", name="")
    trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger)    
    datamodule = AutoDataModule(sequence_length=sequence_length, TrainingDatasetClass=TrainingLSTMDataset).prepare(inputs)
    
    trainer.fit(model=model, datamodule=datamodule)

    #plot training plot    
    inputs_, targets = next(iter(datamodule.train_dataloader(batch_size=datamodule.sizes.train)))
    with torch.no_grad():
        z = model.model(inputs_)
    fig1, ax = plt.subplots()
    for i in range(z.shape[1]):
        ax.plot(targets[:, i].detach(), color='black', label=f'actual {labels[i]}', linestyle='dashed')
        ax.plot(z[:, i].detach(), label=f'training fit {labels[i]}')
    ax.legend()  
    trainer.logger.experiment.add_figure("training", fig1, global_step=trainer.global_step)
    
    #  validation plot
    inputs_, targets = next(iter(datamodule.val_dataloader(batch_size=datamodule.sizes.val)))
    with torch.no_grad():
        z = model.model(inputs_)
    fig2, ax = plt.subplots()
    for i in range(z.shape[1]):
        ax.plot(targets[:, i].detach(), color='black', label=f'actual {labels[i]}', linestyle='dashed') 
        ax.plot(z[:, i].detach(), label=f'validation fit {labels[i]}')
    ax.legend()
    trainer.logger.experiment.add_figure("validation", fig2, global_step=trainer.global_step)
    trainer.logger.experiment.flush()

# evaluate
model.model.eval()
model.model.cpu()
datamodule = PredictLSTMDataset(inputs, sequence_length=sequence_length)

with torch.no_grad():
    for itime in range(sequence_length, datamodule.ntimes):
        inputs = datamodule.get_inputs()
        z = model.model(inputs)
        datamodule.add(z)
        
fig3, ax = plt.subplots()
for i in range(datamodule.predictions.shape[1]):
    ax.plot(datamodule.data[:, i], color='black', label=f'actual {labels[i]}', linestyle='dashed')
    ax.plot(datamodule.predictions[:, i].detach(), label=f'predicted {labels[i]}')
ax.set_xlim(0, 600)
ax.legend()

if train:
    trainer.logger.experiment.add_figure("evaluation", fig3, global_step=trainer.global_step)
    trainer.logger.experiment.flush()
else:
    plt.show()


