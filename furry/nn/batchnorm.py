import furry
import torch

class BatchNormalization(furry.Module):
    def __init__(self, epsilon=0.001, dtype=furry.float32, name="Recurrent", dev=None):
        super(BatchNormalization, self).__init__(input_rank=1, dtype=dtype, name=name, dev=dev)
        self.epsilon = epsilon
        self.init()
    
    def init(self, input_size=None):
        super(BatchNormalization, self).init(input_size)
        self.mean = torch.nn.Parameter(furry.data.scalar(0, dtype=self.dtype), requires_grad=True)
        self.variance = torch.nn.Parameter(furry.data.scalar(1, dtype=self.dtype), requires_grad=True)
        super(BatchNormalization, self)._init_done()
    
    def __logits__(self, x):
        mean = x.mean(0)
        variance = (x - mean).pow(2).mean(0)
        x_h = (x - mean) / (variance + self.epsilon).sqrt()
        return self.mean * x_h + self.variance
