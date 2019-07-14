import torch
from furry.data.utils import float32
from furry.dev import default_device

class Optimizer:
    def __init__(self, module=None, dtype=float32, dev=None):
        if dev is None:
            dev = default_device()
        self.__module = None
        self.__dtype = dtype
        self.__dev = dev
        if module is not None:
            self.module = module
    
    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def device(self):
        return self.__dev

    def step(self):
        pass
    
    def reset_grads(self):
        with torch.no_grad():
            for param in self.module.parameters(recurse=True):
                param.grad.data.zero_()
    
    def init(self):
        pass

    @property
    def module(self):
        return self.__module
    
    @module.setter
    def module(self, module):
        self.__module = module
        self.init()
