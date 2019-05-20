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

default = CPU
if CUDA_AVAILABLE:
    default = gpu(0)
    CUDA_DEVICE_COUNT = torch.cuda.device_count()
