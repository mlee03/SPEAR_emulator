import lightning as pl
from base import SimpleCNN
from auto import AutoTrainModule, AutoDataModule
from pathlib import Path


cnn = SimpleCNN()

reload = False
train = True

if reload:
    latest_ckpt = "/home/Mikyung.Lee/spear-emulator-me/lightning_logs/version_0/checkpoints/epoch=999-step=10000.ckpt"
    model = AutoTrainModule.load_from_checkpoint(latest_ckpt, weights_only=False)
else:
    model = AutoTrainModule(cnn)

data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref")
data.setup()

trainer = pl.Trainer(max_epochs=5000)

if train:
    trainer.fit(model=model, datamodule=data)

def get_mean(data):
    mean = []
    for batch in data:
        mean.extend([idata.deteach().to("cpu").mean() for idata in y[batch]])
    return mean


## evaluate

model.eval()
data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref", testsize=1)
data.setup(training=False)

inputs = data.ds.testing[0]
ntest_times = len(data.ds.testing)
predicted = []
for itime in range(ntest_times):
    z = model(inputs)
    predicted.append(z.detach().to("cpu").item().mean())
    inputs = predicted[-3:]

answers = [data.ds.testing.data[itime].mean() for itime in range(3, ntest_times)]

fig, ax = plt.subplots()
ax.plot(data.ds.testing.time, predicted)
ax.plot(data.ds.testing,time, )



#mean of validated
#y = trainer.predict(model=model, dataloaders=data.val_dataloader)
#val_mean = get_mean(y)
#y = trainer.predict(model=model, dataloaders=data.train_dataloader)
#train_mean = get_mean(y)

#predict
for input in data.