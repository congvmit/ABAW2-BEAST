import os
import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.netmodule import AffWildLightningNet
from utils.datamodule import AffWildDataModule

from tqdm import tqdm

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")

# PERCENT_VALID_EXAMPLES = 0.5
# EPOCHS = 1
DATA_DIR = "/home/lab/congvm/Affwild2"


class Experiment:
    def __init__(self, args):
        self.args = args

    def manual_run(self):

        batch_size = self.args.batch_size
        optimizer_name = self.args.optimizer
        lr = self.args.lr
        model_name = self.args.model_name  #'alternet_18'
        model = AffWildLightningNet(
            model_name=model_name, optimizer_name=optimizer_name, lr=lr
        )
        datamodule = AffWildDataModule(data_dir=DATA_DIR, batch_size=batch_size)

        trainer = pl.Trainer(
            accelerator="gpu",
            logger=True,
            # limit_val_batches=PERCENT_VALID_EXAMPLES,
            enable_checkpointing=True,
            max_epochs=self.args.epochs,
            auto_select_gpus=True,
            gpus=1,
            fast_dev_run=2000,
            # strategy='dp',
            # gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                ModelCheckpoint(
                    filename="{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                    verbose=True,
                    save_top_k=3,
                    mode="min",
                    monitor="val_loss",
                ),
            ],
        )
        hyperparameters = dict(
            model_name=model_name,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            lr=lr,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        print("val_loss:", trainer.callback_metrics["val_loss"].item())
        print("val_acc_exp:", trainer.callback_metrics["val_acc_exp"].item())
        print("val_acc_au:", trainer.callback_metrics["val_acc_au"].item())
        print("val_mse_aro:", trainer.callback_metrics["val_mse_aro"].item())
        print("val_mse_val:", trainer.callback_metrics["val_mse_val"].item())
        print("hp_metric:", trainer.callback_metrics["hp_metric"].item())

    def objective(self, trial: optuna.trial.Trial) -> float:
        # We optimize the number of layers, hidden units in each layer and dropouts.
        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # dropout = trial.suggest_float("dropout", 0.2, 0.5)

        # output_dims = [
        #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
        # ]
        EPOCHS = 10
        batch_size = trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128])
        optimizer_name = trial.suggest_categorical(
            "optimizer_name", choices=["sgd", "adam"]
        )
        lr = trial.suggest_float("lr", 2e-5, 2e-3)
        model_name = "alternet_18"

        model = AffWildLightningNet(
            model_name=model_name, optimizer_name=optimizer_name, lr=lr
        )
        datamodule = AffWildDataModule(data_dir=DATA_DIR, batch_size=batch_size)

        trainer = pl.Trainer(
            accelerator="gpu",
            logger=True,
            # limit_val_batches=PERCENT_VALID_EXAMPLES,
            enable_checkpointing=True,
            max_epochs=self.args.epochs,
            auto_select_gpus=True,
            gpus=1,
            # strategy='dp',
            # gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                ModelCheckpoint(
                    filename="{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                    verbose=True,
                    save_top_k=3,
                    mode="min",
                    monitor="val_loss",
                ),
            ],
        )
        hyperparameters = dict(
            model_name=model_name,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            lr=lr,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics["val_loss"].item()


def debug_dataset():
    BATCH_SIZE = 32
    data_module = AffWildDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    data_module.setup(stage="fit")

    train_dataloader = data_module.train_dataloader()
    for batch in tqdm(train_dataloader):
        pass

    val_dataloader = data_module.val_dataloader()
    for batch in tqdm(val_dataloader):
        pass

    exit()


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("-t", "--timeout", type=int, default=600)
    parser.add_argument("--debug-dataset", action="store_true")
    parser.add_argument("-m", "--manual", action="store_true")

    # Manual
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model-name", type=str, default="alternet_18")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    if args.debug_dataset:
        debug_dataset()

    experiment = Experiment(args)
    if not args.manual:
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner()
            if args.pruning
            else optuna.pruners.NopPruner()
        )

        # This experiment is to minimize val_loss
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(
            experiment.objective,
            n_trials=args.num_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            gc_after_trial=True,
        )

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        experiment.manual_run()
