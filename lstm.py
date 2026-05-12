import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import torch

from model import TrainModule, SimpleLSTM
from data import (
    AutoDataModule, 
    PredictLSTMDataset, 
    TrainingLSTMDataset, 
    load_variable, 
    normalize
)

#lstm parameters
input_size = 1
sequence_length = 5
lstm = SimpleLSTM(input_size=input_size)

#read data
data = load_variable("data/atmos.192101-201012.t_ref.nc", "t_ref")[0]
data = [datum.mean() for datum in data]
data = normalize(data)

reload = True
train = True

max_epochs = 500
learning_rate = 1e-5
saved_chkpt_path = "/home/Mikyung.Lee/spear-emulator-me/lstm-test/version_1/checkpoints/epoch=3999-step=40000.ckpt"


if reload:
    if saved_chkpt_path is None:
        raise ValueError("Set saved_chkpt_path before using reload=True")
    model = TrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = TrainModule(lstm, learning_rate=learning_rate)

if train:
    tb_logger = TensorBoardLogger(save_dir="lstm-test", name="")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
       #fast_dev_run=True
    )
    datamodule = AutoDataModule(sequence_length=sequence_length, TrainingDatasetClass=TrainingLSTMDataset).prepare(data)
    trainer.fit(model=model, datamodule=datamodule)

    #plot training plot    
    inputs, targets = next(iter(datamodule.train_dataloader(batch_size=len(datamodule.train_dataset))))
    with torch.no_grad():
        z = model.model(inputs)
    fig1, ax = plt.subplots()
    ax.plot(targets.detach(), color='black', label='actual')
    ax.plot(z.detach(), color='pink', label='training fit')  
    ax.legend()  

    #  validation plot
    inputs, targets = next(iter(datamodule.val_dataloader(batch_size=len(datamodule.val_dataset))))
    with torch.no_grad():
        z = model.model(inputs)
    fig2, ax = plt.subplots()
    ax.plot(targets.detach(), color='black', label='actual')
    ax.plot(z.detach(), color='pink', label='validation fit')
    ax.legend()


# evaluate
model.model.eval()
model.model.cpu()


#second evaluation
datamodule = PredictLSTMDataset(data, sequence_length=sequence_length)
with torch.no_grad():
    for itime in range(sequence_length, datamodule.ntimes):
        inputs = datamodule.get_inputs()
        z = model.model(inputs)
        datamodule.add(z)
        
fig3, ax = plt.subplots()
ax.plot(datamodule.time, datamodule.data, label='actual')
ax.plot(datamodule.time, datamodule.predictions.detach().numpy(), label='predicted')
ax.legend()

if train:
    if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
        trainer.logger.experiment.add_figure("training", fig1, global_step=trainer.global_step)
        trainer.logger.experiment.add_figure("validation", fig2, global_step=trainer.global_step)
        trainer.logger.experiment.add_figure("evaluation", fig3, global_step=trainer.global_step)
        trainer.logger.experiment.flush()
else:
    plt.show()


