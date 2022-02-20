"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole FashionMNIST dataset, we here use a small subset of it.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]

"""
import argparse
import os
from typing import List
from typing import Optional

from torch._C import dtype

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

import models

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = '/mnt/DATA2/congvm/Affwild2/images/batch'


import models
from models.classifier_block import MTLBlock


class Net(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        cblock = MTLBlock
        if model_name == 'alternet_18':
            self.backbone = models.get_model(model_name, cblock=cblock)
        else:
            self.backbone = models.get_model(model_name, cblock=cblock)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())

from torch.utils import data
import cv2
import numpy as np

class AffWildDataset(data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path):
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        # VideoID	FrameID	Valence	Arousal	Expression	
        # AU1   AU2 AU4	AU6	AU7	AU10	AU12	AU15	AU23	AU24	AU25	AU26
        data = self.df.iloc[index]
        video_id = data.VideoID
        frame_id = data.FrameID

        img_path = os.path.join(*[self.data_dir, video_id, frame_id])
        img_arr = self._load_image(img_path)
        assert img_arr is not None

        if self.transform is not None:
            img_arr = self.transform(img_arr)

        # Valence, Arousal are in range of [-1, 1]
        valence = torch.tensor(float(data.Valence), dtype=torch.float32)
        arousal = torch.tensor(float(data.Arousal), dtype=torch.float32)

        # Exp is integer 
        exp = torch.tensor(int(data.Expression), dtype=torch.long)
        
        # AU is multi-label
        au = torch.from_numpy(np.array(data[5:], dtype=np.float32))

        return {
            'img_arr': img_arr,
            'valence': valence,
            'arousal': arousal,
            'exp': exp,
            'au': au
        }
        
        
class AffWildDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ['fit', 'validate']:
            self.train_ds = AffWildDataset(self.data_dir, os.path.join(self.data_dir, 'mtl_train_anno.csv'))
            self.val_ds = AffWildDataset(self.data_dir, os.path.join(self.data_dir, 'mtl_validation_anno.csv'))
        elif stage in ['predict']:
            raise
        else: 
            #  'test', or 'predict'
            raise
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )


def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)


    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = AffWildDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    # This experiment is to minimize val_loss
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=args.num_trial, timeout=args.timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))