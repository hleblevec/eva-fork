import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Function
import torch
import math

class BinaryQuantizeIRNet(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BiConv2dIRNet(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BiConv2dIRNet, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = nn.Parameter(torch.tensor([10.], requires_grad=True).float())
        self.t = nn.Parameter(torch.tensor([0.1], requires_grad=True).float())

    def forward(self, x):
        w = self.weight
        ba = x
        
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        #bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        constant_tensor = 2 * torch.ones((bw.size(0),), device=x.device, dtype=torch.float)
        sw = torch.pow(constant_tensor, (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        
        k, t = self.k.to(x.device), self.t.to(x.device)
        bw = BinaryQuantizeIRNet().apply(bw, k, t)
        ba = BinaryQuantizeIRNet().apply(ba, k, t)

        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class BiLinearIRNet(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super(BiLinearIRNet, self).__init__(in_channels, out_channels, bias=bias)
        self.k = nn.Parameter(torch.tensor([10.], requires_grad=True).float())
        self.t = nn.Parameter(torch.tensor([0.1], requires_grad=True).float()) 

    def forward(self, x):
        bw = self.weight
        ba = x

        bw = bw - bw.mean(-1).view(-1, 1)
        #bw = bw / bw.std(-1).view(-1, 1)

        constant_tensor = 2 * torch.ones((bw.size(0),), device=x.device, dtype=torch.float)
        sw = torch.pow(constant_tensor, (torch.log(bw.abs().mean(-1)) / math.log(2)).round().float()).view(-1, 1).detach()
        
        k, t = self.k.to(x.device), self.t.to(x.device)
        ba = BinaryQuantizeIRNet().apply(ba, k, t)
        bw = BinaryQuantizeIRNet().apply(bw, k, t)
        
        bw = bw * sw
        output = F.linear(ba, bw, self.bias)
        return output
