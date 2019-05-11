import torch
from furry.data import scalar, upload, float32
from furry.dev import default as default_device

class Optimizer:
    def __init__(self, module=None, dtype=float32, dev=None):
        if dev is None:
            dev = default_device
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

class SGD(Optimizer):
    def __init__(self, module=None, lr=1e-4, momentum=0.9, dev=None):
        super().__init__(module, dev=dev)
        self.lr = lr
        self.momentum = momentum
    
    def init(self):
        self.v = {}
        for param in self.module.parameters(recurse=True):
            self.v[param] = 0

    def step(self):
        with torch.no_grad():
            for param in self.module.parameters(recurse=True):
                grad = param.grad.data.clone()
                m = self.momentum * self.v[param]
                param.data -= m + self.lr * grad
                self.v[param] = grad.detach()

class Adam(Optimizer):
    def __init__(self, module=None, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, dtype=float32, dev=None):
        super().__init__(module, dtype=dtype, dev=dev)
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
    
    def init(self):
        self.m, self.v, self.t = {}, {}, upload(scalar(0.0, dtype=self.dtype), dev=self.device)
        for param in self.module.parameters(recurse=True):
            self.m[param] = upload(scalar(0.0, dtype=self.dtype), dev=self.device)
            self.v[param] = upload(scalar(0.0, dtype=self.dtype), dev=self.device)

    def step(self):
        with torch.no_grad():
            self.t += 1
            for param in self.module.parameters(recurse=True):
                grad = param.grad.data
                self.m[param] = m = self.beta_1 * self.m[param] + (1 - self.beta_1) * grad
                self.v[param] = v = self.beta_2 * self.v[param] + (1 - self.beta_2) * (grad ** 2)
                #m_h = m / (1 - (self.beta_1 ** self.t))
                #v_h = v / (1 - (self.beta_2 ** self.t))
                a_t = self.alpha * torch.sqrt(1 - self.beta_2 ** self.t) / (1 - self.beta_1 ** self.t)
                #param.data = param.data - self.alpha * m_h / (torch.sqrt(v_h) + self.epsilon)
                param.data = param.data - a_t * m / (torch.sqrt(v) + self.epsilon)

