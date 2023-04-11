import torch.nn as nn
import math

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


# only for layer with weight
class LayerConfig:
    def __init__(self, mode=archConst.mode, map_times=1) -> None:
        # 0 for p&n, 1 for ref
        # Todo: pn is not for single layer qzy todo
        # Todo: add weight bit width for single layer, Cheng Huan should modify fixpoint
        if mode == 1:
            self.mode = "ref"
            self.pn_replication_degree = 1
        else:
            self.mode = "pn"
            self.pn_replication_degree = 2

        self.weight_replication_degree = map_times
        self.array_names = []

    def print(self) -> None:
        print("Layer config: ")
        print("    mode: %s." % ("ref" if self.mode == 1 else "p&n"))
        print("    map this layer for %d times." % self.weight_replication_degree)


class PerformanceManager:
    def __init__(self, net: nn.Module, bit_width: int, runMode: archConst.PIMRunMode, map_strategy: MapStrategyBase):
        """
        net: nn.Module
        bit_width: bits of weight
        runMode: running mode
        map_strategy: the strategy of mapping
        """
        self.batch_size = sysPara.batch_size
        self.input_data_shape = sysPara.input_data_shape

        self.bit_width = bit_width
        self.map_strategy = map_strategy
        self.runMode = runMode

        self.layer_record = {}
        self.net_map_times = sysPara.net_map_times

        # map the nn to LogicArray and PhysicalArray
        self.layer_type_list, self.data_shape_list = self.map_network(net)

        # init performance module
        self.areaModule = AreaModule(self.map_strategy)

        # TODO: other module

    def set_layer_config(self, key: str, layer_config: LayerConfig) -> None:
        self.layer_record[key] = layer_config

    def produce_layer_config(self, layer_with_weight_index):
        if self.net_map_times is not None:
            if len(self.net_map_times) <= layer_with_weight_index:
                raise Exception("Illegal length of net map times list! list length: " + str(len(self.net_map_times))
                                + ", layer index: " + str(layer_with_weight_index))

            config = LayerConfig(map_times=self.net_map_times[layer_with_weight_index])
        else:
            config = LayerConfig()

        return config

    def _map_layer_with_weight(self, layer_name: str, layer: nn.Module, layer_index=0) -> None:
        _out, _in = layer.fp_weight.size()
        layer_config = self.produce_layer_config(layer_index)
        self.layer_record[layer_name] = layer_config

        mode, times, map_times = layer_config.mode, layer_config.pn_replication_degree, layer_config.weight_replication_degree

        # map according to given map_strategy
        for i in range(times * map_times):
            array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                format(layer_name=layer_name, mode=mode, modeid=str(i % times), copyid=str(i // times), type="forward")
            self.map_strategy.allocLogicalArray(array_name, [_in, _out], self.bit_width)
            layer_config.array_names.append(array_name)

        # running mode
        if self.runMode != archConst.PIMRunMode.inference:
            # training mode
            for i in range(times * map_times):
                array_name = "{layer_name}_{mode}{modeid}_copy{copyid}_{type}". \
                    format(layer_name=layer_name, mode=mode, modeid=str(i % times), copyid=str(i // times), type="backward")
                self.map_strategy.allocLogicalArray(array_name, [_out, _in], self.bit_width)
                layer_config.array_names.append(array_name)

            # other running mode

    def map_network(self, net):
        index = 0  # real layer
        layer_with_weight_index = 0
        data_shape = self.input_data_shape
        layer_type_list = ["input_" + str(index)]
        data_shape_list = [data_shape]

        for name, module in net.named_modules():
            children_num = sum(1 for _ in module.children())
            if children_num == 0:  # leaf node
                if isinstance(module, fpnn.Linear):
                    index += 1
                    layer_name = "Linear" + "_" + str(index)
                    layer_type_list.append(layer_name)
                    if len(data_shape) != 2 or data_shape[0] != 1 or data_shape[1] != module.in_features:
                        raise Exception("illegal shape, input data shape: " + str(data_shape) + ", but in features: " + str(module.in_features))
                    data_shape = (1, module.out_features)
                    data_shape_list.append(data_shape)
                    # self.layer_record[layer_name] = config
                    # self._map_Linear(layer_name, module, config)
                    self._map_layer_with_weight(layer_name, module, layer_with_weight_index)
                    layer_with_weight_index += 1
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
                    h_out = (h_in + 2 * _padding[0] - _dilation[0] * (_kernel[0] - 1) - 1) // _stride[0] + 1
                    w_out = (w_in + 2 * _padding[1] - _dilation[1] * (_kernel[1] - 1) - 1) // _stride[1] + 1

                    data_shape = (out_channels, h_out, w_out)
                    data_shape_list.append(data_shape)
                    self._map_layer_with_weight(layer_name, module, layer_with_weight_index)
                    layer_with_weight_index += 1
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
                    h_out = (h_in + 2 * _padding - _dilation * (_kernel[0] - 1) - 1) // _stride + 1
                    w_out = (w_in + 2 * _padding - _dilation * (_kernel[1] - 1) - 1) // _stride + 1
                    if channel == 0:
                        data_shape = (h_out, w_out)
                    else:
                        data_shape = (channel, h_out, w_out)

                    data_shape_list.append(data_shape)
                elif isinstance(module, nn.Flatten):
                    # index += 1
                    data_shape = (1, math.prod(data_shape))
                    # layer_name = "Flatten" + "_" + str(index)
                    # layer_type_list.append(layer_name)
                elif not isinstance(module, fpnn.Quan) and not isinstance(module, fpnn.DeQuan):
                    raise Exception("illegal module: " + name)

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
    for layer_name, cfg in performanceManager.layer_record.items():
        print(layer_name)
        print(cfg.array_names)
        for array_name in cfg.array_names:
            print(mapStrategy.getLogicalArrayMap(array_name))

