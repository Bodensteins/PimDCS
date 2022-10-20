from typing import List
from xmlrpc.client import boolean
import torch.nn as nn
import const
import utils
import fixedPoint as fp

from mnist_model import ConvMnist, FcMnist, PimFcMnist


class PerformanceManager:
    def __init__(self, net: nn.Module, h: int, w: int, batch_size: int = 1):
        self.shapes = utils.get_shape(h, w, net)
        self.area_module = AreaModule(net, self.shapes, batch_size)
        self.energy_module = EnergyModule(net)
    
    def print(self):
        self.area_module.print_area_info()
        self.energy_module.print_energy_info()


class AreaModule:
    def __init__(self, net: nn.Module, shapes: List, batch_size: int = 1) -> None:
        self.net = net
        self.shapes = shapes
        self.batch_size = batch_size

        self.PE_size = 0
        # TODO: other device to be considered, according to the architecture
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
                self.PE_size = self.PE_size + self.calc_Linear(layer)

            if isinstance(layer, nn.Conv2d):
                self.PE_size = self.PE_size + self.calc_Conv(idx, layer)

            if isinstance(layer, fp.Linear):
                self.PE_size = self.PE_size + self.calc_fpLinear(layer)

            if isinstance(layer, fp.Conv2d):
                self.PE_size = self.PE_size + self.calc_fpConv(idx, layer)

        return self.PE_size

    def calc_Linear(self, layer: nn.Module) -> int:
        # basic phy array
        _in, _out, _bias = layer.in_features, layer.out_features, layer.bias
        if _bias is not None:
            _in = _in + 1
        m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out, const.unitsPerPhyRow)
        pe_size = utils.ceil(const.times * m * n, const.phyArrayNum)

        if const.runmode != const.PIMRunMode.inference:
            x, y = utils.ceil(_out, const.phyArrRowSize), utils.ceil(_in, const.unitsPerPhyRow)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                row, col = utils.ceil(self.batch_size, const.phyArrRowSize), utils.ceil(m, const.unitsPerPhyRow)
                pe_size = pe_size + utils.ceil(const.times * row * col, const.phyArrayNum)

        return pe_size

    def calc_fpLinear(self, layer: nn.Module) -> int:
        _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
        if _bias == True:
            _in = _in + 1
        m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out, const.unitsPerPhyRow)
        pe_size = utils.ceil(const.times * m * n, const.phyArrayNum)

        if const.runmode != const.PIMRunMode.inference:
            x, y = utils.ceil(_out, const.phyArrRowSize), utils.ceil(_in, const.unitsPerPhyRow)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                row, col = utils.ceil(self.batch_size, const.phyArrRowSize), utils.ceil(m, const.unitsPerPhyRow)
                pe_size = pe_size + utils.ceil(const.times * row * col, const.phyArrayNum)

        return pe_size


    def calc_Conv(self, idx: int, layer: nn.Module) -> int:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.bias
        # choose a better allocation strategy
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias is not None:
            _in = _in + 1
        m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out_channel, const.unitsPerPhyRow)
        pe_size = utils.ceil(const.times * m * n, const.phyArrayNum)

        if const.runmode != const.PIMRunMode.inference:
            x, y = utils.ceil(_kernel[0] * _kernel[1] * _out_channel, const.phyArrRowSize), utils.ceil(_in_channel, const.unitBits)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                shape = self.shapes[idx]
                row, col = utils.ceil(shape[0] * shape[1], const.phyArrRowSize), utils.ceil(_in_channel)
                pe_size = pe_size + self.batch_size * utils.ceil(const.times * row * col, const.phyArrRowSize)

        return pe_size 

    def calc_fpConv(self, idx: int, layer: nn.Module) -> int:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.bias
        # choose a better allocation strategy
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias == True:
            _in = _in + 1
        m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out_channel, const.unitsPerPhyRow)
        pe_size = utils.ceil(const.times * m * n, const.phyArrayNum)

        if const.runmode != const.PIMRunMode.inference:
            x, y = utils.ceil(_kernel[0] * _kernel[1] * _out_channel, const.phyArrRowSize), utils.ceil(_in_channel, const.unitBits)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                shape = self.shapes[idx]
                row, col = utils.ceil(shape[0] * shape[1], const.phyArrRowSize), utils.ceil(_in_channel)
                pe_size = pe_size + self.batch_size * utils.ceil(const.times * row * col, const.phyArrRowSize)

        return pe_size 

class EnergyModule:
    def __init__(self, net: nn.Module):
        self.net = net
        self.epsilon = 1e-5

        self.read_num = 0
        self.write_num = 0
        self.calc_num = 0
        self.read_energy = 0.
        self.write_energy = 0.
        self.calc_energy = 0.

    def print_energy_info(self) -> None:
        print("Energy info:")
        passed = self.check()
        if passed is False:
            return
        print("    read op count: %d" % (self.read_num))
        print("    write op count: %d" % (self.write_num))
        print("    calculation op count: %d" % (self.calc_num))
        print("    memory energy cost: %f" % (self.read_energy + self.write_energy))
        print("    calculation energy cost: %f" % (self.calc_energy))
        print("    total energy cost: %f" % (self.read_energy + self.write_energy + self.calc_energy))

    def get(self, net: nn.Module):
        for idx, (name, layer) in enumerate(self.net.named_modules()):
            pass


    def check(self) -> boolean:
        passed = True

        if const.readUseProbability is True:
            if len(const.CellPD) != const.cellLevels:
                passed =  False
                print("    [Energy] CellPD illegal: the number of probabilities is not cellLevels")
            sum = 0.0
            for p in const.CellPD:
                sum = sum + p
            if abs(sum - 1.0) > self.epsilon:
                passed = False
                print("    [Energy] CellPD illegal: the sum of probabilities is not 1")

        if const.writeUseProbability is True:
            if len(const.writePD) != const.writeParallelism:
                passed = False
                print("    [Energy] writePD illegal: the number of probabilities is not equal to writeParallelism")
            sum = 0.0
            for p in const.writePD:
                sum = sum + p
            if abs(sum - 1.0) > self.epsilon:
                passed = False
                print("    [Energy] writePD illegal: the sum of probabilities is not 1")

        if const.writeUseProbability is True:
            if len(const.inVPD) != const.inVLevels:
                passed = False
                print("    [Energy] inVPD illegal: the number of probabilities is not inVLevels")
            sum = 0.0
            for p in const.writePD:
                sum = sum + p
            if abs(sum - 1.0) > self.epsilon:
                passed = False
                print("    [Energy] inVPD illegal: the sum of probabilities is not 1")

        if passed is True:
            print("    [Energy] params check passed")

        return passed

def test():
    net = PimFcMnist()
    manager = PerformanceManager(net, 1, 784)
    manager.print()

test()