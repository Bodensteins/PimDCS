import math
import torch.nn as nn
import src.pimtorch.nn as fpnn
from functools import reduce

def int_div_ceil(a: int, b: int):
    if a < 0 or b <= 0:
        raise Exception("illegal input, a:" + str(a) + ", b:" + str(b))
    return (a + b - 1) // b


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
        if isinstance(layer, nn.Linear) or isinstance(layer, fpnn.Linear):
            _in, _out = layer.in_features, layer.out_features
            shapes[name] = [1, _out]
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, fpnn.Conv2d):
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


def analyze_network(net, input_data_shape):
    index = 0
    data_shape = input_data_shape
    layer_type_list = ["input_" + str(index)]
    data_shape_list = [data_shape]

    for name, module in net.named_modules():
        children_num = sum(1 for _ in module.children())
        if children_num == 0:  # leaf node
            if isinstance(module, fpnn.Linear):
                index += 1
                layer_type_list.append("Linear" + "_" + str(index))
                if len(data_shape) != 2 or data_shape[0] != 1 or data_shape[1] != module.in_features:
                    raise Exception("illegal shape, input data shape: " + str(data_shape) + ", but in features: " + str(module.in_features))
                data_shape = (1, module.out_features)
                data_shape_list.append(data_shape)
            elif isinstance(module, fpnn.ReLU) or isinstance(module, nn.ReLU):
                index += 1
                layer_type_list.append("ReLU" + "_" + str(index))
                data_shape_list.append(data_shape)
            elif isinstance(module, fpnn.Dropout):
                index += 1
                layer_type_list.append("Dropout" + "_" + str(index))
                data_shape_list.append(data_shape)
            elif isinstance(module, nn.MaxPool2d):
                index += 1
                layer_type_list.append("MaxPool2d" + "_" + str(index))

                _kernel, _padding, _dilation, _stride = module.kernel_size, module.padding, module.dilation, module.stride
                channel, h_in, w_in = 0, 0, 0
                if len(data_shape) == 2:
                    h_in, w_in = data_shape[0], data_shape[1]
                else:
                    channel, h_in, w_in = data_shape[0], data_shape[1], data_shape[2]
                h_out = floor((h_in + 2 * _padding - _dilation * (_kernel[0] - 1) - 1), _stride) + 1
                w_out = floor((w_in + 2 * _padding - _dilation * (_kernel[1] - 1) - 1), _stride) + 1
                if channel == 0:
                    data_shape = (h_out, w_out)
                else:
                    data_shape = (channel, h_out, w_out)

                data_shape_list.append(data_shape)
            elif isinstance(module, fpnn.Conv2d):
                index += 1
                layer_type_list.append("Conv2d" + "_" + str(index))

                _kernel, _padding, _dilation, _stride, out_channels \
                    = module.kernel_size, module.padding, module.dilation, module.stride, module.out_channels

                if len(data_shape) == 2:
                    h_in, w_in = data_shape[0], data_shape[1]
                else:
                    _, h_in, w_in = data_shape[0], data_shape[1], data_shape[2]
                h_out = floor((h_in + 2 * _padding[0] - _dilation[0] * (_kernel[0] - 1) - 1), _stride[0]) + 1
                w_out = floor((w_in + 2 * _padding[1] - _dilation[1] * (_kernel[1] - 1) - 1), _stride[1]) + 1

                data_shape = (out_channels, h_out, w_out)
                data_shape_list.append(data_shape)
            elif isinstance(module, nn.Flatten):
                data_shape = (1, math.prod(data_shape))
                # layer_type_list.append("Flatten")
                pass
            elif not isinstance(module, fpnn.Quan) and not isinstance(module, fpnn.DeQuan):
                raise Exception("illegal module: " + name)

            print("layer index: " + str(index))
            print("layer name: " + name)
            print("layer shape: " + str(data_shape))

    return layer_type_list, data_shape_list


def test():
    pass
    # net = ConvMnist()
    # shapes = get_shape(28, 28, net)