import torch
from furry.optimizer.optimizer import Optimizer
from furry.data import scalar, upload, float32

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
