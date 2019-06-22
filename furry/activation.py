import torch
from furry.func import __cfunc_module

def identity(x):
    return x

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.nn.functional.relu(x)

def leaky_relu(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

def softmax(x, dim=None, dtype=None):
    return torch.nn.functional.softmax(x, dim=dim, dtype=dtype)

Identity = __cfunc_module(identity, "Identity")
Sigmoid = __cfunc_module(sigmoid, "Sigmoid")
Tanh = __cfunc_module(tanh, "Tanh")
Relu = __cfunc_module(relu, "Relu")
LeakyRelu = __cfunc_module(leaky_relu, "LeakyRelu")
Softmax = __cfunc_module(softmax, "Softmax")
