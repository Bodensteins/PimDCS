import torch.nn as nn
from typing import List
import math

from utils import analyze_network, floor
from areaModule import AreaModule
from fixedPoint import nn as fpnn
from mapping.archMapping import MapStrategyBase, DefaultMapStrategy
from mapping.archConst import archConst
from mapping.archRecord import Arch
import systemParameter as sysPara

import sys
sys.path.append("..")
# sys.path.append("../mapping")
from mnist_model import ConvMnist, FcMnist, PimFcMnist, PimConvMnist
 

class LayerConfig:
    def __init__(self, mode=archConst.mode, map_times=1) -> None:
        self.mode = mode  # 0 for p&n, 1 for ref
        self.times = 1 if self.mode == 1 else 2
        self.map_times = map_times
        self.array_names = []

    def print(self) -> None:
        print("Layer config: ")
        print("    mode: %s." % ("ref" if self.mode == 1 else "p&n"))
        print("    map this layer for %d times." % self.map_times)


class PerformanceManager:
    def __init__(self, net: nn.Module, bit_width: int, runMode: archConst.PIMRunMode,
                 map_strategy: MapStrategyBase, net_map_times: List = None):
        """
        net: nn.Module
        input: (batch_size, h, w)
        bit_width: bits of weight
        runMode: running mode
        map_strategy: the strategy of mapping
        """
        # assert(len(input) == 3)
        #
        # batch_size, h, w = input

        # self.batch_size = batch_size
        self.batch_size = sysPara.batch_size
        self.input_data_shape = sysPara.input_data_shape

        self.bit_width = bit_width
        self.map_strategy = map_strategy
        self.runMode = runMode
        # self.shapes, self.layer_names = utils.get_shape(h, w, net)
        # self.layer_type_list, self.data_shape_list = self.analyze_network(net, self.input_data_shape)
        # self.layer_names = self.layer_names[1:]  # input is not layer
        # print(self.layer_type_list)
        # print(self.data_shape_list)

        self.layer_record = {}
        self.net_map_times = net_map_times
        # self.set_default_layer_config(net_map_times)

        self.layer_type_list, self.data_shape_list = self.analyze_network(net, self.input_data_shape)
        self.layer_names = self.layer_type_list[1:]

        # map the nn to LogicArray and PhysicalArray
        # self.map_nn(net)

        # init performance module
        self.areaModule = AreaModule(self.map_strategy)

        # TODO: other module
    #
    # def set_default_layer_config(self, net_map_times):
    #     for i in range(len(self.layer_names)):
    #         key = self.layer_names[i]
    #         if net_map_times is not None:
    #             config = LayerConfig(map_times=net_map_times[i])
    #         else:
    #             config = LayerConfig()
    #         self.layer_record[key] = config

    def set_layer_config(self, key: str, layer_config: LayerConfig) -> None:
        self.layer_record[key] = layer_config

    def map_nn(self, net: nn.Module) -> None:
        '''
        net: nn.Module
        '''

        def map_Linear(idx: int, layer: nn.Module, layer_config: LayerConfig) -> None:
            # shape of logic_array: [_in, _out]
            _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
            if _bias:
                _in = _in + 1

            mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
            mode_name = "ref" if mode == 1 else "pn"

            # map according to given map_strategy
            for i in range(times * map_times):
                array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".\
                    format(id=str(idx - 2), mode=mode_name, modeid=str(i % times), copyid=str(i//times), type="forward")
                self.map_strategy.allocLogicalArray(array_name, [_in, _out], self.bit_width)
                layer_config.array_names.append(array_name)

            # running mode
            if self.runMode != archConst.PIMRunMode.inference:
                # training mode
                for i in range(times * map_times):
                    array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".\
                        format(id=str(idx - 2), mode=mode_name, modeid=str(i % times), copyid=str(i//times),
                               type="backward")
                    self.map_strategy.allocLogicalArray(array_name, [_out, _in], self.bit_width)
                    layer_config.array_names.append(array_name)

                # other running mode

            self.layer_record[str(layer)] = layer_config

        def map_Conv2d(idx, layer: nn.Module, layer_config: LayerConfig) -> None:
            # shape of logic_array: [_in, _out]
            _in_channel, _out_channel, _kernel, _bias \
                = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
            _in = _kernel[0] * _kernel[1] * _in_channel
            if _bias is not None:
                _in = _in + 1

            mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
            mode_name = "ref" if mode == 1 else "pn"

            # map according to given map_strategy
            for i in range(times * map_times):
                array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".\
                    format(id=str(idx - 2), mode=mode_name, modeid=str(i % times), copyid=str(i//times), type="forward")
                self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
                layer_config.array_names.append(array_name)

            # running mode
            if self.runMode != archConst.PIMRunMode.inference:
                # training mode
                _in = _kernel[0] * _kernel[1] * _out_channel
                
                for i in range(times * map_times):
                    array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".\
                            format(id=str(idx - 2), mode=mode_name, modeid=str(i % times), copyid=str(i//times),
                                   type="backward")
                    self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
                    layer_config.array_names.append(array_name)

                # other running mode
            
            self.layer_record[str(layer)] = layer_config

        for idx, (name, layer) in enumerate(net.named_modules()):
            # if you want to duplicate one specific layer, you can map that layer twice or more times
            # 'idx' may help you index that specific layer
            children_num = sum(1 for _ in net.children())
            if children_num == 0:
                if isinstance(layer, fpnn.Linear):
                    map_Linear(idx, layer, self.layer_record[self.layer_names[idx]])

                if isinstance(layer, fpnn.Conv2d):
                    map_Conv2d(idx, layer, self.layer_record[str(layer)])

    def _map_Linear(self, layer_name: str, layer: nn.Module, layer_config: LayerConfig) -> None:
        # shape of logic_array: [_in, _out]
        _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
        if _bias:
            _in = _in + 1

        mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
        mode_name = "ref" if mode == 1 else "pn"

        # map according to given map_strategy
        for i in range(times * map_times):
            array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                format(layer_name=layer_name, mode=mode_name, modeid=str(i % times), copyid=str(i // times), type="forward")
            self.map_strategy.allocLogicalArray(array_name, [_in, _out], self.bit_width)
            layer_config.array_names.append(array_name)

        # running mode
        if self.runMode != archConst.PIMRunMode.inference:
            # training mode
            for i in range(times * map_times):
                array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                    format(layer_name=layer_name, mode=mode_name, modeid=str(i % times), copyid=str(i // times),
                           type="backward")
                self.map_strategy.allocLogicalArray(array_name, [_out, _in], self.bit_width)
                layer_config.array_names.append(array_name)

            # other running mode

        self.layer_record[str(layer)] = layer_config

    def _map_Conv2d(self, layer_name: str, layer: nn.Module, layer_config: LayerConfig) -> None:
        # shape of logic_array: [_in, _out]
        _in_channel, _out_channel, _kernel, _bias \
            = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias is not None:
            _in = _in + 1

        mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
        mode_name = "ref" if mode == 1 else "pn"

        # map according to given map_strategy
        for i in range(times * map_times):
            array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                format(layer_name=layer_name, mode=mode_name, modeid=str(i % times), copyid=str(i // times), type="forward")
            self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
            layer_config.array_names.append(array_name)

        # running mode
        if self.runMode != archConst.PIMRunMode.inference:
            # training mode
            _in = _kernel[0] * _kernel[1] * _out_channel

            for i in range(times * map_times):
                array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                    format(layer_name=layer_name, mode=mode_name, modeid=str(i % times), copyid=str(i // times),
                           type="backward")
                self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
                layer_config.array_names.append(array_name)

            # other running mode

        self.layer_record[str(layer)] = layer_config

    def analyze_network(self, net, input_data_shape):
        index = 0
        data_shape = input_data_shape
        layer_type_list = ["input_" + str(index)]
        data_shape_list = [data_shape]

        for name, module in net.named_modules():
            children_num = sum(1 for _ in module.children())
            if children_num == 0:  # leaf node
                if self.net_map_times is not None:
                    config = LayerConfig(map_times=self.net_map_times[index])
                else:
                    config = LayerConfig()
                if isinstance(module, fpnn.Linear):
                    index += 1
                    layer_name = "Linear" + "_" + str(index)
                    layer_type_list.append(layer_name)
                    if len(data_shape) != 2 or data_shape[0] != 1 or data_shape[1] != module.in_features:
                        raise Exception("illegal shape, input data shape: " + str(data_shape) + ", but in features: " + str(module.in_features))
                    data_shape = (1, module.out_features)
                    data_shape_list.append(data_shape)

                    self.layer_record[layer_name] = config
                    self._map_Linear(layer_name, module, config)
                elif isinstance(module, fpnn.ReLU) or isinstance(module, nn.ReLU):
                    index += 1
                    layer_name = "ReLU" + "_" + str(index)
                    layer_type_list.append(layer_name)
                    data_shape_list.append(data_shape)
                elif isinstance(module, fpnn.Dropout):
                    index += 1
                    layer_name = "Dropout" + "_" + str(index)
                    layer_type_list.append(layer_name)
                    data_shape_list.append(data_shape)
                elif isinstance(module, nn.MaxPool2d):
                    index += 1
                    layer_name = "MaxPool2d" + "_" + str(index)
                    layer_type_list.append(layer_name)

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
                    layer_name = "Conv2d" + "_" + str(index)
                    layer_type_list.append(layer_name)

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

                    self.layer_record[layer_name] = config
                    self._map_Conv2d(layer_name, module, config)
                elif isinstance(module, nn.Flatten):
                    index += 1
                    data_shape = (1, math.prod(data_shape))
                    layer_name = "Flatten" + "_" + str(index)
                    # layer_type_list.append("Flatten")
                    pass
                elif not isinstance(module, fpnn.Quan) and not isinstance(module, fpnn.DeQuan):
                    raise Exception("illegal module: " + name)

                # print("layer index: " + str(index))
                # print("layer name: " + layer_name)
                # print("layer shape: " + str(data_shape))

        return layer_type_list, data_shape_list


if __name__ == "__main__":
    net = PimConvMnist()
    """
    shapes of each layer in this model.
    |-- input                                                                               [28, 28]            
    |-- Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))                                    [10, 24, 24]        
    |-- ReLU()                                                                              [20, 8, 8]          
    |-- MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)     [20, 4, 4]          
    |-- Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))                                   [20, 8, 8]          
    |-- ReLU()                                                                              [20, 8, 8]          
    |-- MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)     [20, 4, 4]          
    |-- Flatten(start_dim=1, end_dim=-1)                                                    [1, 320]            
    |-- Quan()                                                                              [1, 320]            
    |-- Dropout()                                                                           [1, 320]            
    |-- Linear(in_features=320, out_features=10, bias=True)                                 [1, 10]             
    |-- DeQuan()                                                                            [1, 10]   
    """
    print(net)
    ac = Arch(4, 16, 8, 4, 1)
    mapStrategy = DefaultMapStrategy(ac)

    performanceManager = PerformanceManager(net, 8, archConst.PIMRunMode.inference, mapStrategy)
    performanceManager.areaModule.printArch()
    performanceManager.areaModule.print()
    layer_config = performanceManager.layer_record["Conv2d_1"]
    print(layer_config.array_names)

