import torch
import torch.nn as nn
from thirdparty.insightface.recognition.arcface_torch.backbones import iresnet50
import cv2
import torch.nn.functional as F

from numpy.linalg import norm as l2norm


def normed_embedding(embedding):
    return embedding / l2norm(embedding)


class ArcFaceIRes50(nn.Module):
    def __init__(self, ckpt="ckpts/glint360k_cosface_r50_fp16_0.1_backbone.pth"):
        super().__init__()
        self.backbone = iresnet50()
        if ckpt is not None:
            state_dict = torch.load(ckpt)
            print(self.backbone.load_state_dict(state_dict))
        self.out_features = 512

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_feat(self, x):
        with torch.no_grad():
            emb = self(x).detach().cpu().numpy()
        return emb


class MTL_ClassifierGAP(nn.Module):
    def __init__(self, in_features=512):
        super(MTL_ClassifierGAP, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_aro = nn.Linear(in_features, 1)
        self.dense_val = nn.Linear(in_features, 1)
        self.dense_au = nn.Linear(in_features, 12)
        self.dense_exp = nn.Linear(in_features, 8)

        self.fc = nn.Linear(in_features=512, out_features=256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)

        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        x_exp = self.dense_exp(x)
        x_aro = self.dense_aro(x)
        x_val = self.dense_val(x)
        x_au = self.dense_au(x)
        return x_exp, x_aro, x_val, x_au


class MTL_ClassifierMLP(nn.Module):
    def __init__(self, in_features=512, dropout=0.2):
        super(MTL_ClassifierMLP, self).__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.fc_au = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=12),
        )

        self.fc_exp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=8),
        )

        self.fc_aro = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=1),
        )

        self.fc_val = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=1),
        )

    def forward(self, x):
        x_exp = self.fc_exp(x)
        x_aro = self.fc_aro(x)
        x_val = self.fc_val(x)
        x_au = self.fc_au(x)
        return x_exp, x_aro, x_val, x_au


class EXP_ClassifierMLP(nn.Module):
    def __init__(self, in_features=512, dropout=0.2):
        super(EXP_ClassifierMLP, self).__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=8),
        )

    def forward(self, x):
        return self.fc(x)
