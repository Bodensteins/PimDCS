import torch.nn as nn
from typing import List

import sys
sys.path.append("..")
sys.path.append("../mapping")
import utils
from areaModule import AreaModule
from fixedPoint import nn as fpnn
from mapping.archMapping import MapStrategyBase, DefaultMapStrategy
from mapping.archConst import archConst
from mapping.archRecord import Arch
from mnist_model import ConvMnist, FcMnist, PimFcMnist, PimConvMnist


class LayerConfig:
    def __init__(self, mode=archConst.mode, map_times=1) -> None:
        self.mode = mode # 0 for p&n, 1 for ref
        self.times = 1 if self.mode == 1 else 2
        self.map_times = map_times
        self.array_names = []

    def print(self) -> None:
        print("Layer config: ")
        print("    mode: %s." % ("ref" if self.mode == 1 else "p&n"))
        print("    map this layer for %d times." % (self.map_times))
 

class PerformanceManager:
    def __init__(self, net: nn.Module, input: List, bit_width: int, runMode: archConst.PIMRunMode, \
                 map_strategy: MapStrategyBase, net_map_times : List = None):
        """
        net: nn.Module
        input: (batch_size, h, w)
        bit_width: bits of weight
        runMode: running mode
        map_strategy: the strategy of mapping
        """
        assert(len(input) == 3)
        batch_size, h, w = input

        self.batch_size = batch_size
        self.bit_width = bit_width
        self.map_strategy = map_strategy
        self.runMode = runMode
        self.shapes, self.layer_names = utils.get_shape(h, w, net)
        self.layer_names = self.layer_names[1:] # input is not layer 

        self.layer_record = {}
        self.set_default_layer_config(self.shapes, net_map_times)

        # map the nn to LogicArray and PhysicalArray
        self.map_nn(net)

        # init performance module
        self.areaModule = AreaModule(self.map_strategy)

        # TODO: other module

    def set_default_layer_config(self, shapes: dict, net_map_times) -> List:
        for i, (key, _) in enumerate(shapes.items()):
            if net_map_times is not None:
                config = LayerConfig(map_times=net_map_times[i])
            else:
                config = LayerConfig()
            self.layer_record[key] = config
    
    
    def set_layer_config(self, key: str, layer_config: LayerConfig) -> None:
        self.layer_record[key] = layer_config
        

    def map_nn(self, net: nn.Module) -> None:
        '''
        net: nn.Module
        '''

        def map_Linear(idx: int, layer: nn.Module, layer_config: LayerConfig) -> None:
            # shape of logic_array: [_in, _out]
            _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
            if _bias == True:
                _in = _in + 1

            mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
            mode_name = "ref" if mode == 1 else "pn"

            # map according to given map_strategy
            for i in range(times * map_times):
                array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".format(id = str(idx - 2), mode = mode_name, modeid = str(i%times),\
                                                                                  copyid = str(i//times), type="forward")
                self.map_strategy.allocLogicalArray(array_name, [_in, _out], self.bit_width)
                layer_config.array_names.append(array_name)

            # running mode
            if self.runMode != archConst.PIMRunMode.inference:
                # traing mode
                for i in range(times * map_times):
                    array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".format(id = str(idx - 2), mode = mode_name, modeid = str(i%times),\
                                                                                  copyid = str(i//times), type="backward")
                    self.map_strategy.allocLogicalArray(array_name, [_out, _in], self.bit_width)
                    layer_config.array_names.append(array_name)

                # other running mode

            self.layer_record[str(layer)] = layer_config


        def map_Conv2d(idx, layer: nn.Module, layer_config : LayerConfig) -> None:
            # shape of logic_array: [_in, _out]
            _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
            _in = _kernel[0] * _kernel[1] * _in_channel
            if _bias is not None:
                _in = _in + 1

            mode, times, map_times = layer_config.mode, layer_config.times, layer_config.map_times
            mode_name = "ref" if mode == 1 else "pn"

            # map according to given map_strategy
            for i in range(times * map_times):
                array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".format(id = str(idx - 2), mode = mode_name, modeid = str(i%times),\
                                                                                  copyid = str(i//times), type="forward")
                self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
                layer_config.array_names.append(array_name)

            # runing mode
            if self.runMode != archConst.PIMRunMode.inference:
                # traing mode
                _in = _kernel[0] * _kernel[1] * _out_channel
                
                for i in range(times * map_times):
                    array_name = "layer{id}_{mode}{modeid}_copy{copyid}_{type}".format(id = str(idx - 2), mode = mode_name, modeid = str(i%times),\
                                                                                  copyid = str(i//times), type="backward")
                    self.map_strategy.allocLogicalArray(array_name, [_in, _out_channel], self.bit_width)
                    layer_config.array_names.append(array_name)

                # other running mode
            
            self.layer_record[str(layer)] = layer_config

        for idx, (name, layer) in enumerate(net.named_modules()):
            # if you want to duplicate one specific layer, you can map that layer twice or more times
            # 'idx' may help you index that specific layer
            if isinstance(layer, fpnn.Linear):
                map_Linear(idx, layer, self.layer_record[str(layer)])

            if isinstance(layer, fpnn.Conv2d):
                map_Conv2d(idx, layer, self.layer_record[str(layer)])



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

    ac = Arch(4, 16, 8, 4, 1)
    mapStrategy = DefaultMapStrategy(ac)

    performanceManager = PerformanceManager(net, [1, 28, 28], 8, archConst.PIMRunMode.inference, mapStrategy)
    performanceManager.areaModule.printArch()
    performanceManager.areaModule.print()
    layer_config = performanceManager.layer_record[performanceManager.layer_names[3]]
    print(layer_config.array_names)

