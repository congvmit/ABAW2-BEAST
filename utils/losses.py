import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        x = x.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2)))
            * torch.sqrt(torch.sum(torch.pow(vy, 2)))
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = (
            2
            * rho
            * x_s
            * y_s
            / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        )
        return 1 - ccc


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
