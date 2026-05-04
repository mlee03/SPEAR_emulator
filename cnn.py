import lightning as pl
from base import SimpleCNN
from auto import LitAutoCNN, AutoDataModule
from pathlib import Path


cnn = SimpleCNN()
autocnn = LitAutoCNN(cnn)

data = AutoDataModule(datafile="data/atmos.192101-201012.t_ref.nc", variable="t_ref")
data.setup()

trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model=autocnn, datamodule=data)

# Load checkpoint from logdir and run test
ckpt_dir = Path("lightning_logs")
if ckpt_dir.exists():
    # Find the latest version directory
    versions = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("version_")])
    if versions:
        latest_version = versions[-1]
        ckpt_files = list((latest_version / "checkpoints").glob("*.ckpt"))
        if ckpt_files:
            # Load the latest checkpoint
            latest_ckpt = str(sorted(ckpt_files)[-1])
            print(f"Loading checkpoint: {latest_ckpt}")

            model = LitAutoCNN.load_from_checkpoint(latest_ckpt)
            trainer.test(model=model, datamodule=data, ckpt_path=latest_ckpt)
