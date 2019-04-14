import os
import math
import time
import random
import torch
import numpy as np
from furry.dev import default as default_device
from furry.decorators import deprecated
from furry.data import prepend_dimension

def conv2d_output_shape(height, width, filter_height, filter_width, out_channels, stride):
    return (out_channels, ((height - filter_height) / stride + 1), ((width - filter_width) / stride + 1))

def hwc2chw(hwc):
    return hwc.permute(2, 0, 1)

def hwc2bchw(hwc):
    return prepend_dimension(hwc2chw(hwc))

def calc_gain(s, i):
    return math.sqrt((i + s) / (6 * s))

################ Deprecated Methods ################
@deprecated("furry.data.sync_shuffle")
def shuffle_data(x, y):
    seed = round(time.time() * 10e3)
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

@deprecated("furry.data.prepend_dimension")
def add_batch_dimension(tensor):
    return tensor.reshape(1, *tensor.size())

@deprecated("furry.data.to_tensor")
def prepare_data(iterable, dtype):
    if isinstance(iterable, torch.Tensor):
        return iterable
    # .copy() - https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    if isinstance(iterable, np.ndarray):
        return torch.from_numpy(iterable.copy()).type(dtype)
    if isinstance(iterable, list):
        return torch.tensor(iterable, dtype=dtype)

@deprecated("furry.data.stack")
def stack(seq, dim=0):
    return torch.stack(seq, dim=dim)

@deprecated("furry.data.scalar")
def scalar(x):
    if isinstance(x, int):
        return prepare_data([x], torch.int32)
    if isinstance(x, float):
        return prepare_data([x], torch.float32)
    return x

@deprecated("furry.data.upload")
def upload(tensor, dev=default_device):
    return tensor.to(device=dev)

@deprecated("furry.data.download")
def download(tensor):
    return tensor.cpu().detach().numpy()
