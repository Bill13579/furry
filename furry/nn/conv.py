import torch
import furry

class Conv2d(furry.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dtype=furry.float32, dev=None):
        super().__init__(input_rank=3, dtype=dtype, dev=dev)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.init()
    
    def init(self, input_size=None):
        super(Conv2d, self).init(input_size)
        self.weight = torch.nn.Parameter(torch.rand(self.out_channels, int(self.in_channels / self.groups), self.kernel_size[0], self.kernel_size[1], requires_grad=True, dtype=self.dtype))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.rand(self.out_channels, requires_grad=True, dtype=self.dtype))
        else:
            self.bias = None
        super(Conv2d, self)._init_done()
    
    def __logits__(self, x):
        return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

