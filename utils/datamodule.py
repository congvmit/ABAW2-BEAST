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

class AffWildDataset(data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path):
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        # VideoID	FrameID	Valence	Arousal	Expression	
        # AU1   AU2 AU4	AU6	AU7	AU10	AU12	AU15	AU23	AU24	AU25	AU26
        data = self.df.iloc[index]
        video_id = data.VideoID
        frame_id = data.FrameID

        img_path = os.path.join(*[self.data_dir, 'cropped_aligned', video_id, frame_id])
        img_arr = self._load_image(img_path)
        assert img_arr is not None

        if self.transform is not None:
            img_arr = self.transform(image=img_arr)['image']

        # Valence, Arousal are in range of [-1, 1]
        valence = torch.tensor(float(data.Valence), dtype=torch.float32)
        arousal = torch.tensor(float(data.Arousal), dtype=torch.float32)

        # Exp is integer 
        exp = torch.tensor(int(data.Expression), dtype=torch.long)
        
        # AU is multi-label
        au = torch.from_numpy(np.array(data[5:], dtype=np.float32))

        return {
            'img_arr': img_arr,
            'valence': valence,
            'arousal': arousal,
            'exp': exp,
            'au': au
        }
        
        
class AffWildDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int=6):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        # Declare an augmentation pipeline
        
        self.train_transform = A.Compose([
            # A.RandomCrop(width=224, height=224),
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Normalize (mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225), 
                         max_pixel_value=255.0, 
                         always_apply=True),
            ToTensorV2()
        ])
        self.val_test_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize (mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225), 
                         max_pixel_value=255.0, 
                         always_apply=True),
            ToTensorV2()
        ])

        if stage in ['fit', 'validate']:
            self.train_ds = AffWildDataset(self.data_dir, 
                                           os.path.join(self.data_dir, 'mtl_train_anno.csv'),
                                           transform=self.train_transform)
            self.val_ds = AffWildDataset(self.data_dir, 
                                         os.path.join(self.data_dir, 'mtl_validation_anno.csv'), 
                                         transform=self.val_test_transform)
        elif stage in ['predict']:
            raise
        else: #  'test', or 'predict'
            raise
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )