import numpy as np


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
