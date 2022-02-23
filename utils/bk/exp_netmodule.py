import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

import numpy as np
from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MaskedMSELoss
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from .abaw_models import EXP_ClassifierMLP, ArcFaceIRes50

from .metrics import EXP_metric


class EXP_StaticLightningNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # backbone_name, classifier_name, optimizer_name, lr, num_epoch, dropout=0.2, scheduler='constant'
        self.args = args
        self.backbone_name = args.backbone_name
        self.classifier_name = args.classifier_name
        self.scheduler = args.scheduler
        if self.backbone_name == "arcface_ires50":
            self.backbone = ArcFaceIRes50()
        else:
            raise

        if self.classifier_name == "mlp":
            self.classifier = EXP_ClassifierMLP(
                in_features=self.backbone.out_features, dropout=args.dropout
            )
        else:
            raise
        self.loss = CrossEntropyLoss()

        self.optimizer_name = args.optimizer
        self.lr = args.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        img_arr = batch["img_arr"]
        exp = batch["labels"]
        batch_size = len(batch["labels"])

        y_exp = self(img_arr)

        loss = self.loss(input=y_exp, target=exp)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        img_arr = batch["img_arr"]
        exp = batch["labels"]
        batch_size = len(batch["labels"])

        y_exp = self(img_arr)

        loss = self.loss(input=y_exp, target=exp)

        y_exp = torch.softmax(y_exp, dim=1)
        y_exp = y_exp.argmax(dim=1)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return {
            "val_loss": loss.detach(),
            "y_true_exp": exp,
            "y_pred_exp": y_exp,
        }

    def validation_epoch_end(self, outputs):
        val_loss = (
            torch.stack([out["val_loss"] for out in outputs]).mean().cpu().numpy()
        )
        y_true_exp = torch.cat([out["y_true_exp"] for out in outputs]).cpu().numpy()

        y_pred_exp = torch.cat([out["y_pred_exp"] for out in outputs]).cpu().numpy()
        y_pred_exp = y_pred_exp.flatten()

        perf_exp = EXP_metric(y_pred_exp, y_true_exp)
        self.log("val_perf_exp", perf_exp, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            print('>  OPTIMIZER: Load ADAM')
            optimizer= optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
            )
        elif self.optimizer_name == "sgd":
            print('>  OPTIMIZER: Load SGD')
            optimizer =  optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.args.weight_decay
            )
        else:
            raise
        
        if self.scheduler == 'constant':
            return optimizer
        else:
            if self.scheduler == 'cosine':
                print('>  SCHEDULER: Load CosineAnnealingLR')
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(optimizer, self.args.num_epochs, eta_min=1e-4, last_epoch=-1)
            elif self.scheduler == 'warmuplinear':
                pass
            else:
                raise
        return [optimizer], [scheduler]
