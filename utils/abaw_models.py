import torch
import torch.nn as nn
from thirdparty.insightface.recognition.arcface_torch.backbones import iresnet50
import cv2


class ArcFaceIRes50(nn.Module):
    def __init__(self, ckpt="ckpts/glint360k_cosface_r50_fp16_0.1_backbone.pth"):
        super().__init__()
        self.backbone = iresnet50()
        if ckpt is not None:
            state_dict = torch.load(ckpt)
            print(self.backbone.load_state_dict(state_dict))
        self.out_features = 512

    def forward(self, x):
        return self.backbone(x)

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
        self.fc = nn.Linear(in_features=in_features, out_features=in_features // 2)
        self.dense_aro = nn.Linear(in_features // 2, 1)
        self.dense_val = nn.Linear(in_features // 2, 1)
        self.dense_au = nn.Linear(in_features // 2, 12)
        self.dense_exp = nn.Linear(in_features // 2, 8)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc(x)))
        x_exp = self.dense_exp(x)
        x_aro = self.dense_aro(x)
        x_val = self.dense_val(x)
        x_au = self.dense_au(x)
        return x_exp, x_aro, x_val, x_au
