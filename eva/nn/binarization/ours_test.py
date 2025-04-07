import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class BiTestConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(BiTestConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size**2
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        binary_input_no_grad = torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(bw),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(bw)
        cliped_weights = torch.clamp(bw, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BiTestLinear(nn.Linear):
    def __init__(self,  in_channels, out_channels, bias=True):
        super(BiTestLinear, self).__init__(in_channels, out_channels, bias=bias)
 
    def forward(self, input, type=None):
        
        w = self.weight
        bw = w-w.mean(-1).view(-1, 1)
        
        ba = BinaryQuantizer.apply(input)
        bw = BinaryQuantizer.apply(bw)
        
        ba_shape, D = ba.shape[:-1], ba.shape[-1] 
        w_scale = torch.abs(bw).mean(dim=-1, keepdim=True).detach()

        out = nn.functional.linear(ba.view(-1, D), bw * w_scale)
        out = out.view(*ba_shape, -1) 

        return out