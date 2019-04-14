import torch
import furry
from furry.utils import calc_gain
from furry.data import prepend_dimension

class Dense(furry.Module):
    def __init__(self, size, input_size=None, dev=None):
        super(Dense, self).__init__(input_rank=1, dtype=furry.float32, dev=dev)
        self._size = size
        if input_size is not None:
            self.init([input_size])

    def init(self, input_size):
        super(Dense, self).init(input_size)
        gain = calc_gain(self._size, input_size[0])
        self.weight = torch.nn.Parameter(torch.zeros(self._size, input_size[0], requires_grad=True, dtype=self.dtype))
        torch.nn.init.xavier_uniform_(self.weight, gain=gain)
        self.bias = torch.nn.Parameter(torch.zeros(self._size, requires_grad=True, dtype=self.dtype))
        super(Dense, self)._init_done()
    
    def __logits__(self, x):
        return torch.matmul(prepend_dimension(self.weight).repeat(x.size()[0], 1, 1), x.unsqueeze(2)).squeeze(2) + self.bias
