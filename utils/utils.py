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


from torch.nn import Module, ModuleDict
from typing import Iterable
from torch.nn.modules.batchnorm import _BatchNorm


def flatten_modules(modules):
    """This function is used to flatten a module or an iterable of modules into a list of its leaf modules
    (modules with no children) and parent modules that have parameters directly themselves.

    Args:
        modules: A given module or an iterable of modules

    Returns:
        List of modules
    """
    if isinstance(modules, ModuleDict):
        modules = modules.values()

    if isinstance(modules, Iterable):
        _modules = []
        for m in modules:
            _modules.extend(flatten_modules(m))

    else:
        _modules = modules.modules()

    # Capture all leaf modules as well as parent modules that have parameters directly themsleves
    return [m for m in _modules if not list(m.children()) or m._parameters]


def unfreeze_modules(modules):
    """Unfreezes the parameters of the provided modules.

    Args:
        modules: A given module or an iterable of modules
    """
    modules = flatten_modules(modules)
    for module in modules:
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = True


def freeze_modules(modules, train_bn=True):
    """Freezes the parameters of the provided modules.

    Args:
        modules: A given module or an iterable of modules
        train_bn: If True, leave the BatchNorm layers in training mode

    Returns:
        None
    """
    modules = flatten_modules(modules)
    for mod in modules:
        if isinstance(mod, _BatchNorm) and train_bn:
            unfreeze_modules(mod)
        else:
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in mod.parameters(recurse=False):
                param.requires_grad = False
