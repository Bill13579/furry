import torch
import furry.data
import furry.init.heuristic
import furry.activation

strided = torch.strided
sparse_coo = torch.sparse_coo

def zeros(shape, out=None, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return torch.zeros(*shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

def ones(shape, out=None, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return torch.ones(*shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

def rand(shape, out=None, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return 2 * torch.rand(*shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad) - 1

def randn(shape, out=None, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return torch.randn(*shape, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

def he(shape, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return furry.init.heuristic.he(rand(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))

def hen(shape, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return furry.init.heuristic.he(randn(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))

def xavier(shape, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return furry.init.heuristic.xavier(rand(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))

def xaviern(shape, dtype=furry.data.float32, layout=strided, device=furry.data.default_device, requires_grad=False):
    return furry.init.heuristic.xavier(randn(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))
