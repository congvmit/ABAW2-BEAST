import os
import cv2
import numpy as np
import pytorch_lightning as pl
from typing import List
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils import data

import albumentations as A
from albumentations.pytorch import ToTensorV2


from albumentations.core.transforms_interface import ImageOnlyTransform


class ArcFaceTransform(ImageOnlyTransform):
    def __init__(self, input_size, always_apply=True, p=1.0):
        super(ArcFaceTransform, self).__init__(always_apply, p)
        self.input_size = input_size  # (112, 112)
        self.input_mean = 127.5
        self.input_std = 127.5

    def apply(self, img, **params):
        img = cv2.dnn.blobFromImages(
            [img],
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=False,
        )[0]
        return img.transpose((1, 2, 0))

    def get_transform_init_args_names(self):
        return "input_size"


# Transform Example
# self.train_transform = A.Compose(
#     [
#         A.Resize(height=112, width=112),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.ShiftScaleRotate(
#             shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
#         ),
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225),
#             max_pixel_value=255.0,
#             always_apply=True,
#         ),
#         ToTensorV2(),
#     ]
# )

from glob import glob
from tqdm import tqdm


class CSVReader(dict):
    """Faster CSVReader"""

    def __init__(self, df):
        self.df = df
        self.loc = self  # Hack to be the same call as pandas

    def iloc(self, row_idx):
        columns = self.df.columns
        data_dict = {}
        [data_dict.update({col: self.df[col][row_idx]}) for col in columns]
        return data_dict

    def __getitem__(self, index: int):
        return self.iloc(index)

    def __len__(self) -> int:
        return len(self.df)


class Static_AffWildDataset(data.Dataset):
    def __init__(self, data_dir, task, csv_path=None, transform=None):
        self.task = task
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.transform = transform
        if self.task == "mtl":
            assert csv_path is not None
            self.df = pd.read_csv(csv_path)
        else:
            self.list_csv = glob(data_dir + "/*.csv")
            self.df = pd.DataFrame()
            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(
                    drop=True
                )
            self.df = CSVReader(self.df)

    def _load_image(self, img_path):
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.task == "mtl":
            return self.getitem_mtl(index)
        else:
            return self.getitem(index)

    def getitem(self, index):
        data = self.df.iloc(index)
        img_path = data["image_id"]
        img_arr = self._load_image(img_path)

        if self.transform is not None:
            img_arr = self.transform(image=img_arr)["image"]

        if self.task == "exp":
            labels = data["labels_ex"]
        elif self.task == "au":
            labels = np.array(eval(data["labels_au"]), dtype=np.float32)
        elif self.task == "va":
            labels = np.array(data["labels_va"], dtype=np.float32)
        labels = torch.tensor(labels)
        return {"img_arr": img_arr, "labels": labels}

    def getitem_mtl(self, index):
        # VideoID	FrameID	Valence	Arousal	Expression
        # AU1   AU2 AU4	AU6	AU7	AU10	AU12	AU15	AU23	AU24	AU25	AU26
        data = self.df.iloc[index]
        video_id = data.VideoID
        frame_id = data.FrameID

        img_path = os.path.join(*[self.data_dir, "cropped_aligned", video_id, frame_id])
        img_arr = self._load_image(img_path)
        assert img_arr is not None

        if self.transform is not None:
            img_arr = self.transform(image=img_arr)["image"]

        # Valence, Arousal are in range of [-1, 1]
        valence = torch.tensor(float(data.Valence), dtype=torch.float32)
        arousal = torch.tensor(float(data.Arousal), dtype=torch.float32)

        # Exp is integer
        exp = torch.tensor(int(data.Expression), dtype=torch.long)

        # AU is multi-label
        au = torch.from_numpy(np.array(data[5:], dtype=np.float32))

        return {
            "video_id": video_id,
            "frame_id": frame_id,
            "img_arr": img_arr,
            "valence": valence,
            "arousal": arousal,
            "exp": exp,
            "au": au,
        }


class AffWildDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        mode="static",
        task="exp",
        batch_size: int = 32,
        num_workers: int = 6,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.mode = mode
        assert mode in ["static", "sequential"]
        self.task_mapping = {
            "au": "AU_Detection_Challenge",
            "exp": "EXPR_Classification_Challenge",
            "va": "VA_Estimation_Challenge",
        }

    def setup(self, stage: Optional[str] = None) -> None:
        # Declare an augmentation pipeline

        self.train_transform = A.Compose(
            [
                A.Resize(height=112, width=112),
                A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                ArcFaceTransform(input_size=(112, 112)),
                ToTensorV2(),
            ]
        )
        self.val_test_transform = A.Compose(
            [
                A.Resize(height=112, width=112),
                ArcFaceTransform(input_size=(112, 112)),
                ToTensorV2(),
            ]
        )

        if self.mode == "static":
            # data_dir, task, csv_path=None, transform=None
            DataModule = Static_AffWildDataset
        else:
            raise

        if stage in ["fit", "validate"]:
            if self.task == "mtl":
                self.train_ds = DataModule(
                    data_dir=self.data_dir,
                    task=self.task,
                    csv_path=os.path.join(self.data_dir, "mtl_train_anno.csv"),
                    transform=self.train_transform,
                )
                self.val_ds = DataModule(
                    data_dir=self.data_dir,
                    task=self.task,
                    csv_path=os.path.join(self.data_dir, "mtl_validation_anno.csv"),
                    transform=self.val_test_transform,
                )
            else:
                self.train_ds = DataModule(
                    data_dir=os.path.join(
                        *[
                            self.data_dir,
                            "saved_labels",
                            self.task_mapping[self.task],
                            "Train_Set",
                        ]
                    ),
                    task=self.task,
                    transform=self.train_transform,
                )
                self.val_ds = DataModule(
                    data_dir=os.path.join(
                        *[
                            self.data_dir,
                            "saved_labels",
                            self.task_mapping[self.task],
                            "Validation_Set",
                        ]
                    ),
                    task=self.task,
                    transform=self.val_test_transform,
                )
        elif stage in ["predict"]:
            raise
        else:  #  'test', or 'predict'
            raise

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
