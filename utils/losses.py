import torch
import torch.nn as nn


class MaskedBaseLoss(nn.Module):
    def forward(self, input, target, mask):
        denominator = torch.sum(mask)
        if denominator == 0:
            loss = "Ignored"
        else:
            loss = torch.sum(self.loss(input, target * mask) * mask) / torch.sum(mask)
        return loss


class MaskedMSELoss(MaskedBaseLoss):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, input, target, mask):
        mask = mask.view(-1, 1)

        denominator = torch.sum(mask)

        if denominator == 0:
            loss = "Ignored"
        else:
            loss = torch.sum(self.loss(input, target * mask) * mask) / denominator
        return loss


class MaskedCrossEntropyLoss(MaskedBaseLoss):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target, mask):
        denominator = torch.sum(mask)
        if denominator == 0:
            loss = "Ignored"
        else:
            loss = torch.sum(self.loss(input, target * mask) * mask) / denominator
        return loss


class MaskedBCEWithLogitsLoss(MaskedBaseLoss):
    def __init__(self):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
