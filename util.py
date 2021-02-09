import os

import torch
import numpy as np
from PIL import Image

from config.model import max_objects


color_palette = np.loadtxt('palette.txt', dtype=np.uint8).reshape(-1, 3)


def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def convert_to_indexed(array, max_value=max_objects):
    array = (array * torch.arange(1, max_value + 1)[:, None, None]).int()
    array[array == 0] = max_value + 1
    array = array.min(dim=0).values
    array[array == max_value + 1] = 0
    return array


def imwrite_indexed(filename, array):
    array = array.astype(np.uint8)
    """ Save indexed png."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
