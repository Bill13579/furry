import numpy as np

def he(x):
    return x * np.sqrt(2 / x.size()[1])

def xavier(x):
    return x * np.sqrt(1 / x.size()[1])
