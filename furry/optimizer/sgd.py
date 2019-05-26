import torch
from furry.optimizer.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, module=None, lr=1e-4, momentum=0.9, clip=None, dev=None):
        super().__init__(module, dev=dev)
        self.lr = lr
        self.momentum = momentum
        self.clip = clip
    
    @property
    def clip(self):
        return self.__clip
    
    @clip.setter
    def clip(self, clip):
        self.__clip = clip
        if isinstance(self.__clip, float):
            self.__clip = (-self.__clip, self.__clip,)

    def init(self):
        self.v = {}
        for param in self.module.parameters(recurse=True):
            self.v[param] = 0

    def step(self):
        with torch.no_grad():
            for param in self.module.parameters(recurse=True):
                grad = param.grad.data.clone()
                if self.clip is not None:
                    grad = grad.clamp(*self.clip)
                m = self.momentum * self.v[param]
                param.data -= m + self.lr * grad
                self.v[param] = grad.detach()
