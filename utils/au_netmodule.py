from . import models
from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

import numpy as np
from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MaskedMSELoss
from sklearn.metrics import f1_score


def compute_f1_score_au(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(target[:, i], input[:, i])
        f1s.append(f1)
    return np.mean(f1s)


# from models.classifier_block import MTLBlock
class MTLBlock(nn.Module):
    def __init__(self, in_features, num_classes=None, **kwargs):
        super(MTLBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_aro = nn.Linear(in_features, 1)
        self.dense_val = nn.Linear(in_features, 1)
        self.dense_au = nn.Linear(in_features, 12)
        self.dense_exp = nn.Linear(in_features, 8)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x_exp = self.dense_exp(x)
        x_aro = self.dense_aro(x)
        x_val = self.dense_val(x)
        x_au = self.dense_au(x)
        return x_exp, x_aro, x_val, x_au


class SimpleMLP(nn.Module):
    def __init__(self, in_features=512):
        super(SimpleMLP, self).__init__()
        self.in_features = in_features
        self.dense_aro = nn.Linear(in_features, 1)
        self.dense_val = nn.Linear(in_features, 1)
        self.dense_au = nn.Linear(in_features, 12)
        self.dense_exp = nn.Linear(in_features, 8)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        x_exp = self.dense_exp(x)
        x_aro = self.dense_aro(x)
        x_val = self.dense_val(x)
        x_au = self.dense_au(x)
        return x_exp, x_aro, x_val, x_au


class AffWildNet(nn.Module):
    def __init__(self, model_name="alternet_18"):
        super().__init__()
        cblock = MTLBlock
        # model_name == :
        self.backbone = models.get_model(model_name, stem=True, cblock=cblock)
        # else:
        #     self.backbone = models.get_model(model_name, cblock=cblock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_exp, y_aro, y_val, y_au = self.backbone(x)
        return y_exp, y_aro, y_val, y_au


class MTLAffWildLightningNet(pl.LightningModule):
    def __init__(self, model_name, optimizer_name, lr):
        super().__init__()
        self.model_name = model_name
        if model_name == "simplemlp":
            self.model = SimpleMLP()
        else:
            raise ValueError(f"Currently do not support {model_name}")

        # self.model = AffWildNet(model_name)
        self.loss_mse = MaskedMSELoss()
        self.loss_ce = MaskedCrossEntropyLoss()
        self.loss_bce = MaskedBCEWithLogitsLoss()

        self.optimizer_name = optimizer_name
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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

        loss_arousal = self.loss_mse(input=y_aro, target=arousal, mask=arousal != -5)
        loss_valence = self.loss_mse(input=y_val, target=valence, mask=valence != -5)
        loss_expression = self.loss_ce(input=y_exp, target=exp, mask=exp != -1)
        loss_au = self.loss_bce(input=y_au, target=au, mask=au != -1)
        # loss = (loss_arousal + loss_valence + loss_expression + loss_au) / 4

        loss = [
            l
            for l in [loss_arousal, loss_valence, loss_expression, loss_au]
            if l != "Ignored"
        ]
        loss = torch.mean(torch.stack(loss, dim=0))

        # DEBUG
        # import mipkit;mipkit.debug.set_trace();exit();
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
        loss_arousal = self.loss_mse(input=y_aro, target=arousal, mask=arousal != -5)
        loss_valence = self.loss_mse(input=y_val, target=valence, mask=valence != -5)
        loss_expression = self.loss_ce(input=y_exp, target=exp, mask=exp != -1)
        loss_au = self.loss_bce(input=y_au, target=au, mask=au != -1)

        loss = [
            l
            for l in [loss_arousal, loss_valence, loss_expression, loss_au]
            if l != "Ignored"
        ]
        loss = torch.mean(torch.stack(loss, dim=0))

        if torch.isnan(loss):
            # DEBUG
            import mipkit

            mipkit.debug.set_trace()
            exit()

        y_exp = y_exp.argmax(dim=1, keepdim=True)
        y_au = torch.sigmoid(y_au)
        y_au[y_au >= 0.5] = 1.0
        y_au[y_au < 0.5] = 0.0
        # exp_acc = accuracy_score(
        #     exp.detach().cpu().numpy(), y_exp.detach().cpu().numpy()
        # )
        # au_acc = accuracy_score(au.detach().cpu().numpy(), y_au.detach().cpu().numpy())

        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        # self.log("val_acc_exp", exp_acc, on_step=False, on_epoch=True)
        # self.log("val_acc_au", au_acc, on_step=False, on_epoch=True)
        # self.log("val_mse_aro", loss_arousal, on_step=False, on_epoch=True)
        # self.log("val_mse_val", loss_valence, on_step=False, on_epoch=True)
        # self.log("hp_metric", exp_acc, on_step=False, on_epoch=True)

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
        y_true_val = torch.cat([out["y_true_val"] for out in outputs]).cpu().numpy()
        y_true_aro = torch.cat([out["y_true_aro"] for out in outputs]).cpu().numpy()
        y_true_exp = torch.cat([out["y_true_exp"] for out in outputs]).cpu().numpy()
        y_true_au = torch.cat([out["y_true_au"] for out in outputs]).cpu().numpy()

        y_pred_val = torch.cat([out["y_pred_val"] for out in outputs]).cpu().numpy()
        y_pred_aro = torch.cat([out["y_pred_aro"] for out in outputs]).cpu().numpy()
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
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise
