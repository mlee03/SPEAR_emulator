import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import torch

from model import TrainModule, SimpleLSTM
from data import AutoLSTMDataModule, PredictLSTMDataset, load_variable

#lstm parameters
input_size = 1
sequence_length = 5
lstm = SimpleLSTM(input_size=input_size)

#read data
data, ntimes, times = load_variable("data/atmos.192101-201012.t_ref.nc", "t_ref")

reload = False
train = True

max_epochs = 5000
#saved_chkpt_path = "lightning_logs/version_0/checkpoints/epoch=4999-step=50000.ckpt"

if reload:
    model = TrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = TrainModule(lstm)

if train:
    tb_logger = TensorBoardLogger(save_dir="learning-rate-scheduler", name="")
    trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger)
    datamodule = AutoLSTMDataModule(data, sequence_length=sequence_length).setup()
    trainer.fit(model=model, datamodule=datamodule)


# evaluate
model.model.eval()
model.model.cpu()


#first evaluation
datamodule = AutoLSTMDataModule(data, sequence_length=sequence_length, trainingsize=0.999, valsize=0.001).setup()
inputs, targets = next(iter(datamodule.train_dataloader(batch_size=len(datamodule.training_ds))))
with torch.no_grad():
    z = model.model(inputs)
        
fig1, ax = plt.subplots()
ax.plot(targets.detach(), color='black', label='actual')
ax.plot(z.detach(), color='pink', label='fitted')
ax.legend()


#second evaluation
datamodule = PredictLSTMDataset(data, sequence_length=sequence_length)
with torch.no_grad():
    for itime in range(sequence_length, data.ntimes):
        inputs = datamodule.get_inputs()
        z = model.model(inputs)        
        datamodule.add(z)
        
fig2, ax = plt.subplots()
ax.plot(datamodule.time, datamodule.data, label='actual')
ax.plot(datamodule.time, datamodule.predictions.detach().numpy(), label='predicted')
ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('value')

if train:
    if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
        trainer.logger.experiment.add_figure("fits", fig1, global_step=trainer.global_step)
        trainer.logger.experiment.add_figure("predictions", fig2, global_step=trainer.global_step)
        trainer.logger.experiment.flush()
else:
    plt.show()


