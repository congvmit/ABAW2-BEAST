from . import resnet as resnet
from . import resnet_dnn_block as resnet_dnn
from . import resnet_mcdo_block as resnet_mcdo
from . import smoothing_block as smoothing


# Deterministic

def dnn_50(num_classes=10, stem=True, name="resnext_dnn_50", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_101(num_classes=10, stem=True, name="resnext_dnn_101", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout

def mcdo_50(num_classes=10, stem=True, name="resnext_mcdo_50", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_101(num_classes=10, stem=True, name="resnext_mcdo_101", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)

# Deterministic + Smoothing

def dnn_smooth_50(num_classes=10, stem=True, name="resnext_dnn_smoothing_50", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_sblocks=[1, 1, 1, 1],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_101(num_classes=10, stem=True, name="resnext_dnn_smoothing_101", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_sblocks=[1, 1, 1, 1],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_50(num_classes=10, stem=True, name="resnext_mcdo_smoothing_50", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_sblocks=[1, 1, 1, 1],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_101(num_classes=10, stem=True, name="resnext_mcdo_smoothing_101", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_sblocks=[1, 1, 1, 1],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)
