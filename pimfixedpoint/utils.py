import math
from typing import List
import torch.nn as nn
import fixedPoint as fp

from mnist_model import ConvMnist, FcMnist, PimFcMnist

def ceil(x: int, y: int) -> int:
    return int(math.ceil(float(x) / y))

def floor(x: int, y: int) -> int:
    return int(math.floor(float(x) / y))

def get_shape(h: int, w: int, net: nn.Module) -> List:
    shapes = [[h, w]]
    for idx, (name, layer) in enumerate(net.named_modules()):
        input = shapes[-1]
        if isinstance(layer, nn.Linear) or isinstance(layer, fp.Linear):
            _in, _out = layer.in_features, layer.out_features
            shapes.append([1, _out])
        if isinstance(layer, nn.Conv2d) or isinstance(layer, fp.Conv2d):
            _kernel, _padding, _dilation, _stride = layer.kernel_size, layer.padding, layer.dilation, layer.stride
            h_in, w_in = input[0], input[1]
            h_out = floor((h_in + 2 * _padding[0] - _dilation[0] * (_kernel[0] - 1) - 1), _stride[0]) + 1
            w_out = floor((w_in + 2 * _padding[1] - _dilation[1] * (_kernel[1] - 1) - 1), _stride[1]) + 1
            shapes.append([h_out, w_out])
        if isinstance(layer, nn.MaxPool2d):
            # TODO: 2d shape of padding, dilation, stride
            _kernel, _padding, _dilation, _stride = layer.kernel_size, layer.padding, layer.dilation, layer.stride
            h_in, w_in = input[0], input[1]
            h_out = floor((h_in + 2 * _padding - _dilation * (_kernel[0] - 1) - 1), _stride) + 1
            w_out = floor((w_in + 2 * _padding - _dilation * (_kernel[1] - 1) - 1), _stride) + 1
            shapes.append([h_out, w_out])

    return shapes

def test():
    net = ConvMnist()
    shapes = get_shape(28, 28, net)
    print(shapes)
