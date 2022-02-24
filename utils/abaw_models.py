import torch
import torch.nn as nn
from thirdparty.insightface.recognition.arcface_torch.backbones import iresnet50
import cv2
import torch.nn.functional as F

from numpy.linalg import norm as l2norm
from . import backbone


def normed_embedding(embedding):
    return embedding / l2norm(embedding)


import pickle


# ===============================================================================
# **Backbone Layer**
# ===============================================================================


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = backbone.resnet.resnet50()
        pretrained = "ckpts/resnet50-0676ba61.pth"
        state_dict = torch.load(pretrained, map_location="cpu")
        state_dict_ = {}
        for k, v in state_dict.items():
            if "fc" not in k:
                state_dict_[k] = v
        self.backbone.load_state_dict(state_dict_, strict=True)
        self.out_features = 2048

    def forward(self, x):
        return self.backbone(x)


class VGGFaceRes50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = backbone.resnet.resnet50()
        pretrained_vggface2 = "ckpts/resnet50_ft_weight.pkl"
        with open(pretrained_vggface2, "rb") as f:
            pretrained_data = pickle.load(f)

        pretrained_data_ = {}
        for k, v in pretrained_data.items():
            if "fc" not in k:
                pretrained_data_[k] = torch.tensor(v)
        self.backbone.load_state_dict(pretrained_data_, strict=True)
        self.out_features = 2048

    def forward(self, x):
        return self.backbone(x)


class ArcFaceIRes50(nn.Module):
    def __init__(self, ckpt="ckpts/glint360k_cosface_r50_fp16_0.1_backbone.pth"):
        super().__init__()
        self.backbone = iresnet50()
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location="cpu")
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


# ===============================================================================
# **Classification Layer**
# ===============================================================================
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
    def __init__(self, in_features, dropout=0.2):
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


class AU_ClassifierMLP(nn.Module):
    def __init__(self, in_features=512, dropout=0.2):
        super(AU_ClassifierMLP, self).__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features // 2, out_features=12),
        )

    def forward(self, x):
        return self.fc(x)
