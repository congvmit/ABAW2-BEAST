import numpy as np
from sklearn.model_selection import GroupKFold
import pandas as pd


def imdenormalize(
    image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale_255=True
):
    if scale_255:
        image = np.clip(((image * std + mean) * 255.0), 0, 255).astype("uint8")
    else:
        image = image * std + mean
    return image


def impt2np(batch_image, image_index=0):
    image = batch_image[image_index].cpu().numpy().transpose((1, 2, 0))
    return image


def split_data(df_data1, df_data2, number_fold):
    if df_data2 is not None:
        df_data1 = pd.read_csv(df_data1, index_col=False)
        df_data2 = pd.read_csv(df_data2, index_col=False)
        data = pd.concat((df_data1, df_data2), axis=0)
    else:
        data = df_data1

    kf = GroupKFold(n_splits=number_fold)
    df_train = {}
    df_split = {}
    for fold, (train_index, test_index) in enumerate(
        kf.split(data, data, data.iloc[:, 0])
    ):
        df_train[fold] = data.iloc[train_index].reset_index(drop=True)
        df_split[fold] = data.iloc[test_index].reset_index(drop=True)

    return df_train, df_split
