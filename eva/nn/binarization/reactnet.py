import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryQuantizerReactNet(nn.Module):
    def __init__(self):
        super(BinaryQuantizerReactNet, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x) 
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class BiReactNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(BiReactNetConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size) 
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.binary_activation = BinaryQuantizerReactNet()

    def forward(self, x): 
        real_weights = self.weight
        x = self.binary_activation(x) 
        
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BiReactNetLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(BiReactNetLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels)) * 0.001, requires_grad=True)
        self.binary_activation = BinaryQuantizerReactNet()

    def forward(self, x):
        # Apply binary activation to the input
        x = self.binary_activation(x)

        # Get real weights
        real_weights = self.weight

        # Compute the scaling factor
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()  # Detach to stop gradient flow

        # Binary weights without gradients
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)

        # Clipped weights with gradients
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights

        # Perform matrix multiplication
        y = F.linear(x, binary_weights)

        return y