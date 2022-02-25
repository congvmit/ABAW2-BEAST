import os
import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import BackboneFinetuning
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# from utils.mtl_netmodule import MTL_StaticLightningNet
from utils.netmodule import (
    EXP_StaticLightningNet,
    MTL_StaticLightningNet,
    AU_StaticLightningNet,
)

from utils.datamodule import AffWildDataModule

from tqdm import tqdm
from pprint import pprint

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")


class CustomBackboneFinetuning(BackboneFinetuning):
    def freeze_before_training(self, pl_module):
        print("> BackboneFinetuning: Freeze backbone")
        self.freeze(pl_module.backbone)

    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer,
        opt_idx: int,
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            BackboneFinetuning.make_trainable(modules=pl_module.backbone)
            if self.verbose:
                print(f"Current lr: {round(current_lr, self.rounding)}")

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            if self.verbose:
                print(f"Current lr: {round(current_lr, self.rounding)}")


class Experiment:
    def __init__(self, args):
        self.args = args

    def manual_run(self):

        if self.args.mode == "static":
            print("> Load STATIC model")
            if self.args.task == "mtl":
                print("> Load model for MTL task")
                model = MTL_StaticLightningNet(self.args)

                monitor_metric = "val_loss"
                optimal_mode = "min"
            elif self.args.task == "exp":
                print("> Load model for EXP task")
                model = EXP_StaticLightningNet(self.args)

                monitor_metric = "val_perf_exp"
                optimal_mode = "max"

            elif self.args.task == "au":
                print("> Load model for AU task")
                model = AU_StaticLightningNet(self.args)
                monitor_metric = "val_perf_au"
                optimal_mode = "max"
            else:
                raise
        else:
            raise

        datamodule = AffWildDataModule(
            data_dir=self.args.data_dir,
            backbone_name=self.args.backbone_name,
            batch_size=self.args.batch_size,
            mode=self.args.mode,
            task=self.args.task,
        )

        # DEBUG
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=TensorBoardLogger(
                save_dir="./logging",
                name=f"{self.args.task}_{self.args.backbone_name}_{self.args.classifier_name}_{self.args.optimizer}_{self.args.loss}",
            ),
            max_epochs=self.args.num_epochs,
            # auto_select_gpus=True,
            gpus=[self.args.gpu],
            # strategy='dp',
            # gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                ModelCheckpoint(
                    filename="{epoch}-{val_loss:.5f}-{train_loss:.5f}",
                    verbose=True,
                    save_top_k=3,
                    mode=optimal_mode,
                    monitor=monitor_metric,
                ),
                CustomBackboneFinetuning(
                    unfreeze_backbone_at_epoch=args.unfreeze_epoch,
                    verbose=True,
                ),
            ],
        )
        hyperparameters = vars(args)
        pprint(hyperparameters)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)
        # print("val_loss:", trainer.callback_metrics["val_loss"].item())

    def optuna_objective(self, trial: optuna.trial.Trial) -> float:
        raise
        # We optimize the number of layers, hidden units in each layer and dropouts.
        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # dropout = trial.suggest_float("dropout", 0.2, 0.5)

        # output_dims = [
        #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
        # ]
        batch_size = trial.suggest_categorical(
            "batch_size", choices=[16, 32, 64, 128, 256, 512]
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer_name", choices=["sgd", "adam"]
        )
        lr = trial.suggest_float("lr", 2e-5, 2e-3)
        backbone_name = self.args.backbone_name
        classifier_name = self.args.classifier_name

        model = MTL_StaticLightningNet(
            backbone_name=backbone_name,
            classifier_name=classifier_name,
            optimizer_name=optimizer_name,
            lr=lr,
        )
        datamodule = AffWildDataModule(
            data_dir=self.args.data_dir, batch_size=batch_size
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            logger=True,
            max_epochs=self.args.num_epochs,
            # auto_select_gpus=True,
            gpus=str(self.args.gpu),
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
            backbone_name=backbone_name,
            classifier_name=classifier_name,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            lr=lr,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        pprint(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics["val_loss"].item()


def debug_dataset(args):
    data_module = AffWildDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        mode=args.mode,
        task=args.task,
        num_workers=args.num_workers,
    )
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
    parser.add_argument("-a", "--auto", action="store_true")

    parser.add_argument("-g", "--gpu", type=int, default=0)

    # DATA_DIR = "/home/lab/congvm/Affwild2"
    parser.add_argument("--data-dir", type=str, default="/home/lab/congvm/Affwild2")
    parser.add_argument(
        "--task",
        type=str,
        default="exp",
        choices=["exp", "au", "mtl", "va"],
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="arcface_ires50",
        choices=["arcface_ires50", "fecnet", "resnet50", "vggresnet50", "dan"],
    )
    parser.add_argument(
        "--classifier-name",
        type=str,
        default="mlp",
        choices=["mlp"],
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument("--unfreeze-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument(
        "--loss", type=str, default="default", choices=["default", "focal"]
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "constant"]
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)

    parser.add_argument(
        "--mode", type=str, default="static", choices=["static", "sequential"]
    )
    # Manual
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=30)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    if args.debug_dataset:
        debug_dataset(args)

    experiment = Experiment(args)
    if args.auto:
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner()
            if args.pruning
            else optuna.pruners.NopPruner()
        )

        # This experiment is to minimize val_loss
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(
            experiment.optuna_objective,
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
