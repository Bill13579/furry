import torch

CUDA_AVAILABLE = torch.cuda.is_available()

device = torch.device

def cpu():
    return device("cpu")

def gpu(i):
    return device("cuda:" + str(i))

def gpu_device_name(dev):
    return torch.cuda.get_device_name(dev)

CPU = cpu()

__INITIAL_DEFAULT = CPU
if CUDA_AVAILABLE:
    __INITIAL_DEFAULT = gpu(0)
    CUDA_DEVICE_COUNT = torch.cuda.device_count()

__DEFAULT = __INITIAL_DEFAULT

def default_device():
    return __DEFAULT

def set_default_device(d=__INITIAL_DEFAULT):
    global __DEFAULT
    __DEFAULT = d
    return __DEFAULT
