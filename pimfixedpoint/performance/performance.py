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

class PerformanceManager:
    def __init__(self, net: nn.Module, input: List, bit_width: int, runMode: archConst.PIMRunMode, map_strategy: MapStrategyBase):
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
        self.shapes = utils.get_shape(h, w, net)

        # map the nn to LogicArray and PhysicalArray
        self.map_nn(net)

        # init performance module
        self.areaModule = AreaModule(self.map_strategy)

        # TODO: other module
        

    def map_nn(self, net: nn.Module) -> None:
        '''
        net: nn.Module
        '''
        def map_Linear(name: str, layer: nn.Module) -> None:
            # shape of logic_array: [_in, _out]
            _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
            if _bias == True:
                _in = _in + 1

            # map according to given map_strategy
            for i in range(archConst.times):
                self.map_strategy.allocLogicalArray(name + "inference" + str(i), [_in, _out], self.bit_width)

            # running mode
            if self.runMode != archConst.PIMRunMode.inference:
                # traing mode
                for i in range(archConst.times):
                    self.map_strategy.allocLogicalArray(name + "training" + str(i), [_out, _in], self.bit_width)

                # other running mode

        def map_Conv2d(name: str, layer: nn.Module) -> None:
            # shape of logic_array: [_in, _out]
            _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
            _in = _kernel[0] * _kernel[1] * _in_channel
            if _bias is not None:
                _in = _in + 1

            # map according to given map_strategy
            for i in range(archConst.times):
                self.map_strategy.allocLogicalArray(name + "inference" + str(i), [_in, _out_channel], self.bit_width)

            # runing mode
            if self.runMode != archConst.PIMRunMode.inference:
                # traing mode
                _in = _kernel[0] * _kernel[1] * _out_channel
                for i in range(archConst.times):
                    self.map_strategy.allocLogicalArray(name + "training" + str(i), [_in, _out_channel], self.bit_width)

                # other running mode

        for idx, (name, layer) in enumerate(net.named_modules()):
            # if you want to duplicate one specific layer, you can map that layer twice or more times
            # 'idx' may help you index that specific layer
            if isinstance(layer, fpnn.Linear):
                map_Linear(name, layer)

            if isinstance(layer, fpnn.Conv2d):
                map_Conv2d(name, layer)


if __name__ == "__main__":
    net = PimConvMnist()
    ac = Arch(4, 16, 8, 4, 1)
    mapStrategy = DefaultMapStrategy(ac)

    performanceManager = PerformanceManager(net, [1, 28, 28], 8, archConst.PIMRunMode.inference, mapStrategy)
    performanceManager.areaModule.printArch()
    performanceManager.areaModule.print()
