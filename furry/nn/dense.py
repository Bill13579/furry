import torch
import furry
from furry.utils import calc_gain
from furry.data import prepend_dimension

class Dense(furry.Module):
    def __init__(self, size, input_size=None, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, dtype=furry.float32, name="Dense", dev=None):
        super(Dense, self).__init__(input_rank=1, dtype=dtype, name=name, dev=dev)
        self._size = size
        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        if input_size is not None:
            self.init([input_size])

    def init(self, input_size):
        super(Dense, self).init(input_size)
        self.weight = torch.nn.Parameter(self.weight_initialization([self._size, input_size[0]], requires_grad=True, dtype=self.dtype))
        self.bias = torch.nn.Parameter(self.bias_initialization([self._size], requires_grad=True, dtype=self.dtype))
        super(Dense, self)._init_done()
    
    def __logits__(self, x):
        return torch.matmul(prepend_dimension(self.weight).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias
