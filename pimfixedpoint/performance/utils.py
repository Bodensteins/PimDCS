import math
from typing import List
from functools import reduce
from torch import dropout
import torch.nn as nn

import sys
sys.path.append("..")
import fixedPoint as fp
from mnist_model import ConvMnist, FcMnist, PimFcMnist


def ceil(x: int, y: int) -> int:
    return int(math.ceil(float(x) / y))


def floor(x: int, y: int) -> int:
    return int(math.floor(float(x) / y))


def get_shape(h: int, w: int, net: nn.Module):
    shapes = {"input": [h, w]}
    names = ["input"]
    for idx, (name, layer) in enumerate(net.named_modules()):
        if idx == 0 or idx == 1: continue
        input = shapes[names[-1]]
        names.append(str(layer))
        name = names[-1]
        if isinstance(layer, nn.Linear) or isinstance(layer, fp.Linear):
            _in, _out = layer.in_features, layer.out_features
            shapes[name] = [1, _out]
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, fp.Conv2d):
            _kernel, _padding, _dilation, _stride, out_channels = layer.kernel_size, layer.padding, layer.dilation, layer.stride, layer.out_channels
            if len(input) == 2:
                h_in, w_in = input[0], input[1]
            else:
                channel, h_in, w_in = input[0], input[1], input[2]
            h_out = floor(
                (h_in + 2 * _padding[0] - _dilation[0] * (_kernel[0] - 1) - 1), _stride[0]) + 1
            w_out = floor(
                (w_in + 2 * _padding[1] - _dilation[1] * (_kernel[1] - 1) - 1), _stride[1]) + 1
            shapes[name] = [out_channels, h_out, w_out]

        elif isinstance(layer, nn.MaxPool2d):
            # TODO: 2d shape of padding, dilation, stride
            _kernel, _padding, _dilation, _stride = layer.kernel_size, layer.padding, layer.dilation, layer.stride
            channel, h_in, w_in = 0, 0, 0
            if len(input) == 2:
                h_in, w_in = input[0], input[1]
            else:
                channel, h_in, w_in = input[0], input[1], input[2]
            h_out = floor((h_in + 2 * _padding - _dilation *
                          (_kernel[0] - 1) - 1), _stride) + 1
            w_out = floor((w_in + 2 * _padding - _dilation *
                          (_kernel[1] - 1) - 1), _stride) + 1
            if channel == 0:
                shapes[name] = [h_out, w_out]
            else:
                shapes[name] = [channel, h_out, w_out]
        elif isinstance(layer, nn.Flatten):
            shapes[name] = [1, reduce(lambda x, y: x * y, input)]
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Dropout):
            shapes[name] = input
        else:
            shapes[name] = input
    print("shapes of each layer in this model.")
    mat = "{:4}{:80}\t{:20}"
    for name in names:
        print(mat.format("|-- ", name, str(shapes[name])))

    return shapes, names


def test():
    net = ConvMnist()
    shapes = get_shape(28, 28, net)