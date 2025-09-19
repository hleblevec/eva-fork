import torch
from torch import nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity, TruncAvgPool2d

from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, IntBias, TruncTo8bit


class CommonIntWeightPerChannelQuant(Int8WeightPerTensorFloat):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloat):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloat):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = RestrictValueType.LOG_FP



class BrevitasQuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(BrevitasQuantConv2d, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            bias_quant=IntBias,
            weight_bit_width=4, 
            weight_quant=CommonIntWeightPerChannelQuant
        )

    def forward(self, x):
        return self.conv(x)

class BrevitasQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BrevitasQuantLinear, self).__init__()
        self.linear = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            bias_quant=IntBias,
            weight_bit_width=4,
            weight_quant=CommonIntWeightPerChannelQuant,
        )

    def forward(self, x):
        return self.linear(x)

class BrevitasQuantReLU(nn.Module):
    def __init__(self):
        super(BrevitasQuantReLU, self).__init__()
        self.relu = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=4,
            return_quant_tensor=True,
        )

    def forward(self, x):
        return self.relu(x)

class BrevitasTruncAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(BrevitasTruncAvgPool2d, self).__init__()
        self.pool = TruncAvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bit_width=4,
            return_quant_tensor=True,
            trunc_quant=TruncTo8bit
        )

    def forward(self, x):
        return self.pool(x)