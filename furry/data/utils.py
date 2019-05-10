import time
import random
import torch
import furry
from furry.dev import default as default_device
import numpy as np

torch.set_printoptions(precision=15)

float32 = torch.float32
float64 = torch.float64
float16 = torch.float16
uint8 = torch.uint8
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
Tensor = torch.Tensor
tensor = torch.tensor

def prepend_dimension(x):
    return x.reshape(1, *x.size())

def append_dimension(x):
    return x.reshape(*x.size(), 1)

def cast(x, dtype=float32):
    return x.to(dtype)

def to_tensor(iterable, dtype=float32):
    if isinstance(iterable, torch.Tensor):
        return iterable
    # .copy() - https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    if isinstance(iterable, np.ndarray):
        return torch.from_numpy(iterable.copy()).type(dtype)
    if isinstance(iterable, list):
        return tensor(iterable, dtype=dtype)
    raise TypeError("iterable must be of type torch.Tensor, np.ndarray, or list")

def scalar(x, dtype=None):
    if dtype is not None:
        return to_tensor([x], dtype=dtype)
    if isinstance(x, int):
        return to_tensor([x], int32)
    if isinstance(x, float):
        return to_tensor([x], float32)
    raise TypeError("x must be of type int or float")

def upload(x, dev=default_device):
    return x.to(device=dev)

def download(x):
    return upload(x, dev=furry.dev.CPU).detach().numpy()

def stack(seq, dim=0):
    return torch.stack(seq, dim=dim)

def unstack(x, dim=0):
    return torch.unbind(x, dim=dim)

def sync_shuffle(*xs, seed=None):
    if seed is None:
        seed = round(time.time() * 10e3)
    for x in xs:
        random.seed(seed)
        random.shuffle(x)
