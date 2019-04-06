import math
import torch
import furry
from .dense import Dense
import numpy as np
from furry.utils import add_batch_dimension, calc_gain

class GRUGate(Dense):
    def __logits__(self, x):
        return furry.activation.sigmoid(super().__logits__(x))

class GRU(furry.Module):
    def __init__(self, size, input_size=None, dev=None):
        super(GRU, self).__init__(dtype=furry.float32, dev=dev)
        self.memory = None
        self._size = size
        if input_size is not None:
            self.init([input_size])
    
    def clear_memory(self):
        self.memory = None

    def init(self, input_size):
        super(GRU, self).init(input_size)
        gain = calc_gain(self._size, input_size[0])
        self.weight = torch.nn.Parameter(torch.zeros(self._size, input_size[0], requires_grad=True, dtype=self.dtype))
        torch.nn.init.xavier_uniform_(self.weight, gain=gain)
        self.weight_h = torch.nn.Parameter(torch.zeros(self._size, self._size, requires_grad=True, dtype=self.dtype))
        torch.nn.init.xavier_uniform_(self.weight_h, gain=gain)
        self.bias = torch.nn.Parameter(torch.zeros(self._size, requires_grad=True, dtype=self.dtype))
        self.bias_h = torch.nn.Parameter(torch.zeros(self._size, requires_grad=True, dtype=self.dtype))
        self.gates = {
            "update": GRUGate(1, input_size=self._size)
        }
        super(GRU, self)._init_done()
    
    def __broadcast__(self, x):
        if len(x.size()) == 1:
            x = furry.utils.add_batch_dimension(x)
        if len(x.size()) == 2:
            x = x.unsqueeze(2)
        return x
    
    def __single(self, x):
        return torch.matmul(add_batch_dimension(self.weight).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias

    def __hidden(self, x):
        return torch.matmul(add_batch_dimension(self.weight_h).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias_h

    def __logits__(self, x, return_all=True, hidden_state=None):
        time_steps = torch.unbind(x, 2)
        last_ts_out = hidden_state
        if last_ts_out is None:
            last_ts_out = 0
        results = []
        for ts in time_steps:
            ts_out = self.__single(ts)
            if last_ts_out != 0:
                ts_out += self.__hidden(last_ts_out)
            update_gate_out = self.gates["update"](ts_out)
            last_ts_out = update_gate_out * ts_out + (1 - update_gate_out) * last_ts_out
            results.append(last_ts_out)
        self.memory = last_ts_out.detach()
        return last_ts_out if not return_all else torch.stack(results, dim=2)
