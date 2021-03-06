import torch
import furry

class _ConvTransposeNd(furry.Module):
    def __init__(self, d, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, dtype=furry.float32, name=None, dev=None):
        super().__init__(input_rank=d+1, dtype=dtype, name=name, dev=dev)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.weight_initialization = weight_initialization
        self.bias_initialization = bias_initialization
        self.init()
    
    def init(self, input_size=None):
        super(_ConvTransposeNd, self).init(input_size)
        self.weight = torch.nn.Parameter(self.weight_initialization([self.in_channels, int(self.out_channels / self.groups), *self.kernel_size], requires_grad=True, dtype=self.dtype))
        if self.use_bias:
            self.bias = torch.nn.Parameter(self.bias_initialization([self.out_channels], requires_grad=True, dtype=self.dtype))
        else:
            self.bias = None
        super(_ConvTransposeNd, self)._init_done()

class ConvTranspose1d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, dtype=furry.float32, name="Conv1d", dev=None):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        super().__init__(1, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, weight_initialization=weight_initialization, bias_initialization=bias_initialization, dtype=dtype, name=name, dev=dev)

    def __logits__(self, x):
        return torch.nn.functional.conv_transpose1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class ConvTranspose2d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, dtype=furry.float32, name="Conv2d", dev=None):
        super().__init__(2, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, weight_initialization=weight_initialization, bias_initialization=bias_initialization, dtype=dtype, name=name, dev=dev)

    def __logits__(self, x):
        return torch.nn.functional.conv_transpose2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class ConvTranspose3d(_ConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_initialization=furry.init.rand, bias_initialization=furry.init.zeros, dtype=furry.float32, name="Conv3d", dev=None):
        super().__init__(3, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, weight_initialization=weight_initialization, bias_initialization=bias_initialization, dtype=dtype, name=name, dev=dev)

    def __logits__(self, x):
        return torch.nn.functional.conv_transpose3d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

DeConv1d = ConvTranspose1d
DeConv2d = ConvTranspose2d
DeConv3d = ConvTranspose3d
