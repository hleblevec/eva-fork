import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as init

from .xilinx_attn import SelfAttention

from .quantizer.quant_brevitas import BrevitasQuantConv2d, BrevitasQuantLinear, BrevitasQuantReLU, BrevitasTruncAvgPool2d
from .mp import MPConv2d, MPLinear, mp_sum
from ..binarization.dorefa import DoReFaConv2d, DoReFaLinear 
from ..binarization.irnet import BiLinearIRNet, BiConv2dIRNet 
from ..binarization.ours_test import BiTestLinear, BiTestConv2d 
from ..binarization.reactnet import BiReactNetLinear, BiReactNetConv2d 

xilinx_leaky_relu_neg_slope = 0.1015625

# Conv Block
conv_types = {
    "normal": nn.Conv2d,
    "mp": MPConv2d,
    "dorefa_binarization": DoReFaConv2d,
    "irnet_binarization": BiConv2dIRNet,
    "ours_binarization" : BiTestConv2d, 
    "reactnet_binarization": BiReactNetConv2d,
    "brevitas" : BrevitasQuantConv2d,
}
    
    
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: dict, bias: bool=False):
        super().__init__()
        self.method = conv_types[config["block_method"]]
        
        self.conv1 = ConvBlock(in_dim, out_dim, config, stride=1, bias=bias, activation=nn.Identity()) 
        #self.conv2 = ConvBlock(out_dim, out_dim, config, stride=1, bias=bias)  
        
        if in_dim != out_dim:  
            raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == MPConv2d:
            return mp_sum(self.shortcut(x), self.conv2(self.conv1(x)))
        return x +  self.conv1(x) #self.conv2(self.conv1(x))
        
    
class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: dict, stride: int = 2, padding: int=1, bias = None, 
                 activation: nn.Module=nn.LeakyReLU(xilinx_leaky_relu_neg_slope)):
        super().__init__()
        
        method = config["block_method"]
        precision = config["precision"] 
        
        self.conv = conv_types[method](  
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )  

        self.conv_block = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(out_dim),   
            activation, 
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) 

# Linear Block
lin_types = {
    "normal": nn.Linear,
    "mp": MPLinear,
    "dorefa_binarization": DoReFaLinear,
    "irnet_binarization": BiLinearIRNet,
    "ours_binarization" : BiTestLinear, 
    "reactnet_binarization": BiReactNetLinear,
    "brevitas" : BrevitasQuantLinear
}

class LinearBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: dict,
                 activation: nn.Module = nn.LeakyReLU(xilinx_leaky_relu_neg_slope), bias = None):
        super().__init__()

        method = config["block_method"]
        precision = config["precision"] 
        
        self.linear = lin_types[method](in_dim, out_dim, bias=bias)

        layers = [self.linear]
        if activation:
            layers.append(activation) 

        self.linear_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_block(x)


#AvgPoolingBlock
pool_types = {
    "normal": nn.AvgPool2d,
    "brevitas": BrevitasTruncAvgPool2d,
}

class AvgPoolingBlock(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, config: dict):
        super().__init__()
        
        method = config["block_method"]
        self.pool = pool_types[method](kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.attn_block = nn.Sequential(
            SelfAttention(in_dim),  
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn_block(x)