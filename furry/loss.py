import torch

def l1(y, gt):
    return torch.sum(torch.abs(gt - y))

def l2(y, gt):
    return torch.sum((gt - y) ** 2)

def mse(y, gt):
    return torch.mean((gt - y) ** 2)

def cross_entropy(y, gt):
    return torch.mean(-torch.sum(gt * torch.log(y), 1))
