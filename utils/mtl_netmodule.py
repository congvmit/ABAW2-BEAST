import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

import numpy as np
from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MaskedMSELoss
from sklearn.metrics import f1_score
from .abaw_models import MTL_ClassifierMLP, ArcFaceIRes50


def compute_f1_score_au(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(target[:, i], input[:, i])
        f1s.append(f1)
    return np.mean(f1s)


class MTL_StaticLightningNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.backbone_name = args.backbone_name
        self.classifier_name = args.classifier_name
        if self.backbone_name == "arcface_ires50":
            self.backbone = ArcFaceIRes50()
        else:
            raise

        if self.classifier_name == "mlp":
            self.classifier = MTL_ClassifierMLP(
                in_features=self.backbone.out_features, dropout=args.dropout
            )
        else:
            raise
        self.loss_mse = MaskedMSELoss()
        self.loss_ce = MaskedCrossEntropyLoss()
        self.loss_bce = MaskedBCEWithLogitsLoss()

        self.optimizer_name = args.optimizer_name
        self.lr = args.lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x)

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
        perf_au = compute_f1_score_au(y_pred_au, y_true_au)
        self.log("val_perf_au", perf_au, prog_bar=True, on_step=False, on_epoch=True)

        # VA
        # TODO: Add VA metrics

        # MTL
        # TODO: Add VA metrics

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer_name == "adam":
            return optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
            )
        elif self.optimizer_name == "sgd":
            return optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
            )
        else:
            raise
