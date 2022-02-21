from . import models
from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MaskedMSELoss

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


class AffWildLightningNet(pl.LightningModule):
    def __init__(self, model_name, optimizer_name, lr):
        super().__init__()
        self.model = AffWildNet(model_name)
        self.loss_mse = MaskedMSELoss()
        self.loss_ce = MaskedCrossEntropyLoss()
        self.loss_bce = MaskedBCEWithLogitsLoss()

        self.optimizer_name = optimizer_name
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        img_arr = batch["img_arr"]
        valence = batch["valence"].view(-1, 1)
        arousal = batch["arousal"].view(-1, 1)
        exp = batch["exp"]
        au = batch["au"]

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

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        img_arr = batch["img_arr"]
        valence = batch["valence"].view(-1, 1)
        arousal = batch["arousal"].view(-1, 1)
        exp = batch["exp"]
        au = batch["au"]

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
        exp_acc = accuracy_score(
            exp.detach().cpu().numpy(), y_exp.detach().cpu().numpy()
        )
        au_acc = accuracy_score(au.detach().cpu().numpy(), y_au.detach().cpu().numpy())
        # exp_accuracy = y_exp.eq(exp.view_as(y_exp)).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc_exp", exp_acc, on_step=False, on_epoch=True)
        self.log("val_acc_au", au_acc, on_step=False, on_epoch=True)
        self.log("val_mse_aro", loss_arousal, on_step=False, on_epoch=True)
        self.log("val_mse_val", loss_valence, on_step=False, on_epoch=True)
        self.log("hp_metric", exp_acc, on_step=False, on_epoch=True)

        return {"val_loss": loss.detach()}

    # def validation_epoch_end(self, outputs):
    #     # DEBUG
    #     import mipkit

    #     mipkit.debug.set_trace()
    #     exit()

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise
