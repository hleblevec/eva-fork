import numpy as np
import torch 
import torch.nn.functional as F
import torch.nn as nn
import pdb

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.
def weight_normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    alpha=np.sqrt(norm.numel() / x.numel())
    return x /  torch.add(eps, norm, alpha=alpha)

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).
class mp_silu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).
def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).
def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

class MPConv2d(nn.Conv2d):
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight)*(1 / np.sqrt(fan_in))
        return F.conv2d(x, weight, None, padding=self.padding, stride=self.stride)


class MPLinear(nn.Linear):
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(weight_normalize(self.weight))
        fan_in = self.weight[0].numel()
        weight = weight_normalize(self.weight)*(1 / np.sqrt(fan_in))
        return F.linear(x, weight, None)
