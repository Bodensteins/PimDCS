import torch.nn as nn
import torch
import const
import utils

from mnist_model import ConvMnist, FcMnist, PimFcMnist

class AreaModule:
    def __init__(self, net: nn.Module) -> None:
        self.net = net

        self.PE_size = 0
        
        self.adc_area = const.single_adc_area * const.phyArrColSize * const.phyArrayNum / const.adc_shared_ratio
        self.dac_area = const.single_dac_area * const.phyArrRowSize * const.phyArrayNum / const.dac_shared_ratio
        self.SH_area = const.single_SH_area * const.phyArrColSize * const.phyArrayNum
        self.array_area = const.single_cell_area * const.phyArrRowSize * const.phyArrColSize * const.phyArrayNum 
        self.adder_area = const.adder_area
        self.pe_area = self.array_area + self.adc_area + self.dac_area + self.adder_area + self.SH_area

    def print_area_info(self) -> None:
        self.PE_size = self.get_PE_size()
        print("Area info:")
        print("    pe_size : %d" % self.PE_size)
        print("    physical array area: %f mm^2" % (self.array_area * self.PE_size / 1e6))
        print("    dac area: %f mm^2" % (self.dac_area * self.PE_size / 1e6))
        print("    adc array area: %f mm^2" % (self.adc_area * self.PE_size / 1e6))
        print("    SH array area: %f mm^2" % (self.SH_area * self.PE_size / 1e6))
        print("    adder array area: %f mm^2" % (self.adder_area * self.PE_size / 1e6))
        print("    total area: %f mm^2" % (self.pe_area * self.PE_size / 1e6))

    def get_PE_size(self) -> int:
        for idx, (name, layer) in enumerate(self.net.named_modules()):
            if isinstance(layer, nn.Linear):
                # basic phy array
                _in, _out, _bias = layer.in_features, layer.out_features, layer.bias
                m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out, const.unitsPerPhyRow)
                if _bias is not None:
                    m = m + 1
                self.PE_size = self.PE_size + utils.ceil(const.times * m * n, const.phyArrayNum)

                # run mode
                # TODO
                # if const.runmode != const.PIMRunMode.inference:
                #     x, y = utils.ceil(_out, const.phyArrRowSize), utils.ceil(_in, const.unitsPerPhyRow)
                #     self.PE_size = self.PE_size + utils.ceil(const.times * x * y, const.phyArrayNum)

                #     if const.runmode == const.PIMRunMode.train_transientInBuffer:
                #         pass
                #     else:
                #         pass # TODO

            if isinstance(layer, nn.Conv2d):
                _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.bias
                # choose a better allocation strategy
                m, n = _kernel[0] * _kernel[1] * _in_channel, _out_channel
                if _bias is not None:
                    m = m + 1
                self.PE_size = self.PE_size + utils.ceil(const.times * m * n, const.phyArrayNum)

        return self.PE_size
            

def test():
    net = ConvMnist()
    areaModule = AreaModule(net)
    areaModule.print_area_info()

test()