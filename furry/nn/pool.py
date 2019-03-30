import torch
import furry

class _MaxPoolNd(furry.Module):
    def __init__(self, d, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, dev=None):
        super().__init__(input_rank=d+1, dtype=furry.float32, dev=dev)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.to(self.device)

class MaxPool1d(_MaxPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, dev=None):
        super().__init__(1, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class MaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, dev=None):
        super().__init__(2, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class MaxPool3d(_MaxPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, dev=None):
        super().__init__(3, kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.max_pool3d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class _AvgPoolNd(furry.Module):
    def __init__(self, d, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, dev=None):
        super().__init__(input_rank=d+1, dtype=furry.float32, dev=dev)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.to(self.device)

class AvgPool1d(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, dev=None):
        super().__init__(1, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.avg_pool1d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad)

class AvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, dev=None):
        super().__init__(2, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.avg_pool2d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad)

class AvgPool3d(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, dev=None):
        super().__init__(3, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, dev=dev)

    def __logits__(self, input):
        return torch.nn.functional.avg_pool3d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad)
