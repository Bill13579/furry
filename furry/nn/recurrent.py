import math
import torch
import furry
import numpy as np
from furry.utils import calc_gain
from furry.data import prepend_dimension

class Recurrent(furry.Module):
    def __init__(self, size, input_size=None, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, weight_h_initialization=furry.init.rand, bias_h_initialization=furry.init.zeros, dtype=furry.float32, name="Recurrent", dev=None):
        super(Recurrent, self).__init__(dtype=dtype, name=name, dev=dev)
        self.memory = None
        self._size = size
        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.weight_h_initialization = weight_h_initialization
        self.bias_h_initialization = bias_h_initialization
        if input_size is not None:
            self.init([input_size])
    
    def clear_memory(self):
        self.memory = None

    def init(self, input_size):
        super(Recurrent, self).init(input_size)
        self.weight = torch.nn.Parameter(self.weight_initialization([self._size, input_size[0]], requires_grad=True, dtype=self.dtype))
        self.weight_h = torch.nn.Parameter(self.weight_h_initialization([self._size, self._size], requires_grad=True, dtype=self.dtype))
        self.bias = torch.nn.Parameter(self.bias_initialization([self._size], requires_grad=True, dtype=self.dtype))
        self.bias_h = torch.nn.Parameter(self.bias_h_initialization([self._size], requires_grad=True, dtype=self.dtype))
        super(Recurrent, self)._init_done()
    
    def __broadcast__(self, x):
        if len(x.size()) == 1:
            x = prepend_dimension(x)
        if len(x.size()) == 2:
            x = x.unsqueeze(2)
        return x
    
    def __single(self, x):
        return torch.matmul(prepend_dimension(self.weight).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias

    def __hidden(self, x):
        return torch.matmul(prepend_dimension(self.weight_h).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias_h

    def __logits__(self, x, return_all=True, hidden_state=None):
        time_steps = torch.unbind(x, 2)
        last_ts_out = hidden_state
        results = []
        for ts in time_steps:
            ts_out = self.__single(ts)
            if last_ts_out is not None:
                ts_out += self.__hidden(last_ts_out)
            last_ts_out = ts_out
            results.append(ts_out)
        self.memory = last_ts_out.detach()
        return last_ts_out if not return_all else torch.stack(results, dim=2)
