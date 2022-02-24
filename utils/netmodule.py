import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import numpy as np

from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from .abaw_models import EXP_ClassifierMLP, AU_ClassifierMLP, MTL_ClassifierMLP
from .backbone.FECNet import FECNet
from .metrics import EXP_metric

from .abaw_models import VGGFaceRes50, ArcFaceIRes50, ResNet50
from .metrics import AU_metric
from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MaskedMSELoss


def get_backbone(backbone_name):
    if backbone_name == "arcface_ires50":
        backbone = ArcFaceIRes50()
    elif backbone_name == "fecnet":
        backbone = FECNet(pretrained=True)
    elif backbone_name == "vggresnet50":
        backbone = VGGFaceRes50()
    elif backbone_name == "resnet50":
        backbone = ResNet50()
    else:
        raise
    return backbone


def get_classifier(classifier_name, task, out_features, args):
    if classifier_name == "mlp":
        if task == "au":
            classifier = AU_ClassifierMLP(
                in_features=out_features, dropout=args.dropout
            )
        elif task == "exp":
            classifier = EXP_ClassifierMLP(
                in_features=out_features, dropout=args.dropout
            )
        elif task == "mtl":
            classifier = MTL_ClassifierMLP(
                in_features=out_features, dropout=args.dropout
            )
    else:
        raise
    return classifier


# Basemodel
class BaseStaticLightningNet(pl.LightningModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)

    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            print(">  OPTIMIZER: Load ADAM")
            optimizer = optim.Adam(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            print(">  OPTIMIZER: Load SGD")
            optimizer = optim.SGD(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise

        if self.args.scheduler == "constant":
            return optimizer
        else:
            if self.args.scheduler == "cosine":
                print(">  SCHEDULER: Load CosineAnnealingLR")
                from torch.optim.lr_scheduler import CosineAnnealingLR

                scheduler = CosineAnnealingLR(
                    optimizer, self.args.num_epochs, eta_min=1e-4, last_epoch=-1
                )
            elif self.args.scheduler == "warmuplinear":
                pass
            else:
                raise
        return [optimizer], [scheduler]


# Expression
class EXP_StaticLightningNet(BaseStaticLightningNet):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = get_backbone(args.backbone_name)
        self.classifier = get_classifier(
            args.classifier_name, args.task, self.backbone.out_features, args
        )
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        img_arr = batch["img_arr"]
        exp = batch["labels"]
        batch_size = len(batch["labels"])

        y_exp = self(img_arr)
        loss = self.loss(input=y_exp, target=exp)

        y_exp = torch.softmax(y_exp, dim=1)
        y_exp = y_exp.argmax(dim=1)
        perf_exp = EXP_metric(y_exp.detach().cpu().numpy(), exp.detach().cpu().numpy())
        self.log(
            "perf_exp",
            perf_exp,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
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


# Action Unit Classification
from torch.nn import BCEWithLogitsLoss
from .metrics import AU_metric
from .abaw_models import AU_ClassifierMLP


class AU_StaticLightningNet(BaseStaticLightningNet):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = get_backbone(args.backbone_name)
        self.classifier = get_classifier(
            args.classifier_name, args.task, self.backbone.out_features, args
        )
        self.loss = BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        img_arr = batch["img_arr"]
        au = batch["labels"]
        batch_size = len(batch["labels"])

        y_au = self(img_arr)

        loss = self.loss(input=y_au, target=au)
        y_au = torch.sigmoid(y_au)
        y_au[y_au >= 0.5] = 1.0
        y_au[y_au < 0.5] = 0.0
        perf_au = AU_metric(y_au.detach().cpu().numpy(), au.detach().cpu().numpy())
        self.log(
            "perf_au",
            perf_au,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        img_arr = batch["img_arr"]
        au = batch["labels"]
        batch_size = len(batch["labels"])

        y_au = self(img_arr)

        loss = self.loss(input=y_au, target=au)
        y_au = torch.sigmoid(y_au)
        y_au[y_au >= 0.5] = 1.0
        y_au[y_au < 0.5] = 0.0
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
            "y_true_au": au,
            "y_pred_au": y_au,
        }

    def validation_epoch_end(self, outputs):
        val_loss = (
            torch.stack([out["val_loss"] for out in outputs]).mean().cpu().numpy()
        )
        y_true_au = torch.cat([out["y_true_au"] for out in outputs]).cpu().numpy()
        y_pred_au = torch.cat([out["y_pred_au"] for out in outputs]).cpu().numpy()

        perf_au = AU_metric(y_pred_au, y_true_au)
        self.log("val_perf_au", perf_au, prog_bar=True, on_step=False, on_epoch=True)


# Multi-task
class MTL_StaticLightningNet(BaseStaticLightningNet):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args.backbone_name)
        self.classifier = get_classifier(
            args.classifier_name, args.task, self.backbone.out_features, args
        )
        self.loss_mse = MaskedMSELoss()
        self.loss_ce = MaskedCrossEntropyLoss()
        self.loss_bce = MaskedBCEWithLogitsLoss()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        video_id = batch["video_id"]
        frame_id = batch["frame_id"]
        img_arr = batch["img_arr"]
        valence = batch["valence"].view(-1, 1)
        arousal = batch["arousal"].view(-1, 1)
        exp = batch["exp"]
        au = batch["au"]
        batch_size = len(batch["video_id"])

        y_exp, y_aro, y_val, y_au = self(img_arr)

        # loss_arousal = self.loss_mse(input=y_aro, target=arousal, mask=arousal != -5)
        # loss_valence = self.loss_mse(input=y_val, target=valence, mask=valence != -5)
        loss_expression = self.loss_ce(input=y_exp, target=exp, mask=exp != -1)
        loss_au = self.loss_bce(input=y_au, target=au, mask=au != -1)
        # loss = (loss_arousal + loss_valence + loss_expression + loss_au) / 4

        loss = [l for l in [loss_expression, loss_au] if l != "Ignored"]
        loss = torch.mean(torch.stack(loss, dim=0))
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        video_id = batch["video_id"]
        frame_id = batch["frame_id"]
        img_arr = batch["img_arr"]
        valence = batch["valence"].view(-1, 1)
        arousal = batch["arousal"].view(-1, 1)
        exp = batch["exp"]
        au = batch["au"]
        batch_size = len(batch["video_id"])

        y_exp, y_aro, y_val, y_au = self(img_arr)

        # DEBUG
        # loss_arousal = self.loss_mse(input=y_aro, target=arousal, mask=arousal != -5)
        # loss_valence = self.loss_mse(input=y_val, target=valence, mask=valence != -5)
        loss_expression = self.loss_ce(input=y_exp, target=exp, mask=exp != -1)
        loss_au = self.loss_bce(input=y_au, target=au, mask=au != -1)

        loss = [l for l in [loss_expression, loss_au] if l != "Ignored"]
        loss = torch.mean(torch.stack(loss, dim=0))

        y_exp = y_exp.argmax(dim=1, keepdim=True)
        y_au = torch.sigmoid(y_au)
        y_au[y_au >= 0.5] = 1.0
        y_au[y_au < 0.5] = 0.0
        # exp_acc = accuracy_score(
        #     exp.detach().cpu().numpy(), y_exp.detach().cpu().numpy()
        # )
        # au_acc = accuracy_score(au.detach().cpu().numpy(), y_au.detach().cpu().numpy())

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
            "video_id": video_id,
            "frame_id": frame_id,
            "y_true_val": valence,
            "y_true_aro": arousal,
            "y_true_exp": exp,
            "y_true_au": au,
            "y_pred_val": y_val,
            "y_pred_aro": y_aro,
            "y_pred_exp": y_exp,
            "y_pred_au": y_au,
        }

    def validation_epoch_end(self, outputs):
        val_loss = (
            torch.stack([out["val_loss"] for out in outputs]).mean().cpu().numpy()
        )
        # y_true_val = torch.cat([out["y_true_val"] for out in outputs]).cpu().numpy()
        # y_true_aro = torch.cat([out["y_true_aro"] for out in outputs]).cpu().numpy()
        y_true_exp = torch.cat([out["y_true_exp"] for out in outputs]).cpu().numpy()
        y_true_au = torch.cat([out["y_true_au"] for out in outputs]).cpu().numpy()

        # y_pred_val = torch.cat([out["y_pred_val"] for out in outputs]).cpu().numpy()
        # y_pred_aro = torch.cat([out["y_pred_aro"] for out in outputs]).cpu().numpy()
        y_pred_exp = torch.cat([out["y_pred_exp"] for out in outputs]).cpu().numpy()
        y_pred_exp = y_pred_exp.flatten()
        y_pred_au = torch.cat([out["y_pred_au"] for out in outputs]).cpu().numpy()

        video_ids = []
        frame_ids = []
        for out in outputs:
            frame_ids.extend(out["frame_id"])
            video_ids.extend(out["video_id"])

        perf_exp = f1_score(
            y_pred_exp, y_true_exp, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], average="macro"
        )
        self.log("val_perf_exp", perf_exp, prog_bar=True, on_step=False, on_epoch=True)

        # AU
        perf_au = AU_metric(y_pred_au, y_true_au)
        self.log("val_perf_au", perf_au, prog_bar=True, on_step=False, on_epoch=True)

        # VA
        # TODO: Add VA metrics

        # MTL
        # TODO: Add VA metrics
