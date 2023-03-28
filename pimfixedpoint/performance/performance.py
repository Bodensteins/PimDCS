from typing import List
import torch.nn as nn
from performance import const, utils
import fixedPoint as fp

from mnist_model import ConvMnist


class PerformanceManager:
    def __init__(self, net: nn.Module, h: int, w: int, batch_size: int = 1):
        self.shapes = utils.get_shape(h, w, net)
        self.area_module = AreaModule(net, self.shapes, batch_size)
        self.energy_module = EnergyModule(net, self.shapes, batch_size)
        self.latency_module = LatencyModule(net, self.shapes, batch_size)
    
    def print(self):
        self.area_module.print_area_info()
        self.energy_module.print_energy_info()
        self.latency_module.print_latency_info()


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
            x, y = utils.ceil(_kernel[0] * _kernel[1] * _out_channel, const.phyArrRowSize), utils.ceil(_in_channel, const.unitsPerPhyRow)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                shape = self.shapes[idx]
                row, col = utils.ceil(self.batch_size * shape[0] * shape[1], const.phyArrRowSize), utils.ceil(_in_channel, const.unitsPerPhyRow)
                pe_size = pe_size + utils.ceil(const.times * row * col, const.phyArrayNum)

        return pe_size 

    def calc_fpConv(self, idx: int, layer: nn.Module) -> int:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
        # choose a better allocation strategy
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias == True:
            _in = _in + 1
        m, n = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out_channel, const.unitsPerPhyRow)
        pe_size = utils.ceil(const.times * m * n, const.phyArrayNum)

        if const.runmode != const.PIMRunMode.inference:
            x, y = utils.ceil(_kernel[0] * _kernel[1] * _out_channel, const.phyArrRowSize), utils.ceil(_in_channel, const.unitsPerPhyRow)
            pe_size = pe_size + utils.ceil(const.times * x * y, const.phyArrayNum)

            if const.runmode != const.PIMRunMode.train_transientInBuffer:
                shape = self.shapes[idx]
                row, col = utils.ceil(self.batch_size * shape[0] * shape[1], const.phyArrRowSize), utils.ceil(_in_channel, const.unitsPerPhyRow)
                pe_size = pe_size + utils.ceil(const.times * row * col, const.phyArrayNum)

        return pe_size 


class EnergyModule:
    def __init__(self, net: nn.Module, shapes: List, batch_size: int = 1) -> None:
        self.net = net
        self.batch_size = batch_size
        self.shapes = shapes
        self.epsilon = 1e-5

        self.adc_energy = 0.
        self.dac_energy = 0.
        self.read_energy = 0.
        self.write_energy = 0.
        self.mm_energy = 0.
        self.calc_energy = 0.
        
        passed = self.check()
        if passed is False:
            pass
        self.set_params()
        self.get_average_write_energy()

    def set_params(self) -> None:
        if const.readUseProbability is True:
            self.cell_pd = const.CellPD
        else:
            if const.CellPDDefault == 0:
                self.cell_pd = const.cellLevels * [1.0 / const.cellLevels]
            else:
                self.cell_pd = const.cellLevels * [0.0]
                self.cell_pd[-1] = 1.0

        if const.writeUseProbability is True:
            self.write_pd = const.writePD
        else:
            if const.writePDDefault == 0:
                self.cell_pd = const.writeParallelism * [1.0 / const.writeParallelism]
            else:
                self.cell_pd = const.writeParallelism * [0.0]
                self.cell_pd[-1] = 1.0

        if const.computeUseProbability is True:
            self.inV_pd = const.inVPD
        else:
            if const.inVPDDefault == 0:
                self.cell_pd = const.inVLevels * [1.0 / const.inVLevels]
            else:
                self.cell_pd = const.inVLevels * [0.0]
                self.cell_pd[-1] = 1.0

    def print_energy_info(self) -> None:
        self.get_info(self.net)
        print("Energy info(nJ):")
        print("    read energy cost: %f" % (self.read_energy))
        print("    write energy cost: %f" % (self.write_energy))
        print("    memory energy cost: %f" % (self.read_energy + self.write_energy))
        print("    crossbar compute energy cost: %f" % (self.mm_energy))
        print("    adc energy cost: %f" % (self.adc_energy))
        print("    dac energy cost: %f" % (self.dac_energy))
        print("    compute energy cost: %f" % (self.calc_energy))
        print("    total energy cost: %f" % (self.read_energy + self.write_energy + self.calc_energy))

    def get_info(self, net: nn.Module) -> None:
        for idx, (name, layer) in enumerate(self.net.named_modules()):
            if isinstance(layer, nn.Linear):
                res = self.calc_Linear(layer)
                self.read_energy = self.read_energy + res[0]
                self.write_energy = self.write_energy + res[1]
                self.mm_energy = self.mm_energy + res[2]

            if isinstance(layer, nn.Conv2d):
                res = self.calc_Conv(idx, layer)
                self.read_energy = self.read_energy + res[0]
                self.write_energy = self.write_energy + res[1]
                self.mm_energy = self.mm_energy + res[2]

            if isinstance(layer, fp.Linear):
                res = self.calc_fpLinear(layer)
                self.read_energy = self.read_energy + res[0]
                self.write_energy = self.write_energy + res[1]
                self.mm_energy = self.mm_energy + res[2]

            if isinstance(layer, fp.Conv2d):
                res = self.calc_fpConv(idx, layer)
                self.read_energy = self.read_energy + res[0]
                self.write_energy = self.write_energy + res[1]
                self.mm_energy = self.mm_energy + res[2]

    # forward:  1. write input matrix (training mode)
    #           2. mm (input X weight)
    # backward: 1. get grad_input  -> mm (grad_output X weight.t())
    #           2. get grad_weight -> mm (input.t() X grad_output)
    # update:   1. write weight.t()
    #           2. read new weight
    #           3. write weight for transpose
    def calc_Linear(self, layer: nn.Module) -> List[float]:
        _in, _out, _bias = layer.in_features, layer.out_features, layer.bias
        if _bias is not None:
            _in = _in + 1
        row, col = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out, const.unitsPerPhyRow)
        # forward
        read_energy = 0.
        write_energy = self.get_energy_per_write(1, _in) * self.batch_size
        mm_energy = self.get_energy_per_mm(row, col) * self.batch_size
        # backward
        mm_energy = mm_energy + self.get_energy_per_mm(col, row) * self.batch_size
        mm_energy = mm_energy + self.get_energy_per_mm(_in, 1) * self.batch_size

        # update (once)
        write_energy = write_energy + self.get_energy_per_write(_out, _in)
        read_energy = read_energy + self.get_energy_per_read(_out, utils.ceil(_in, const.phyArrColSize) * const.phyArrColSize)
        write_energy = write_energy + self.get_energy_per_write(_in, _out)
        return [read_energy, write_energy, mm_energy]

    def calc_fpLinear(self, layer: nn.Module) -> List[float]:
        _in, _out, _bias = layer.in_features, layer.out_features, layer.hasBias
        if _bias == True:
            _in = _in + 1
        row, col = utils.ceil(_in, const.phyArrRowSize), utils.ceil(_out, const.unitsPerPhyRow)
        # forward
        read_energy = 0.
        write_energy = self.get_energy_per_write(1, _in) * self.batch_size
        mm_energy = self.get_energy_per_mm(row, col) * self.batch_size
        # backward
        mm_energy = mm_energy + self.get_energy_per_mm(col, row) * self.batch_size
        mm_energy = mm_energy + self.get_energy_per_mm(_in, 1) * self.batch_size

        # update (once a batch)
        write_energy = write_energy + self.get_energy_per_write(_out, _in)
        read_energy = read_energy + self.get_energy_per_read(_out, utils.ceil(_in, const.phyArrColSize) * const.phyArrColSize)
        write_energy = write_energy + self.get_energy_per_write(_in, _out)
        return [read_energy, write_energy, mm_energy]

    # forward:  1. write input matrix (training mode)
    #           2. mm (unfold(input) X weight)
    # backward: 1. get grad_input  -> Full-Conv(rot180(weight), grad_output)
    #           2. get grad_weight -> Conv(input, grad_output)
    #           TIPS: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
    # update:   1. write weight
    #           2. read new weight
    #           3. write weight for transpose 
    def calc_Conv(self, idx: int, layer: nn.Module) -> List[float]:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.bias
        # choose a better allocation strategy
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias is not None:
            _in = _in + 1
        input_shape = self.shapes[idx]
        output_shape = self.shapes[idx + 1]
        unfold_L = output_shape[0] * output_shape[1]
        # forward
        write_energy = self.get_energy_per_write(input_shape[0] * input_shape[1], _in_channel) * self.batch_size
        mm_energy = self.get_energy_per_mm(_in, _out_channel) * unfold_L * self.batch_size 
        # backward
        mm_energy = mm_energy + self.get_energy_per_mm(_out_channel, _in) * input_shape[0] * input_shape[1] * self.batch_size
        mm_energy = mm_energy + self.get_energy_per_mm(input_shape[0] * input_shape[1], _in_channel) * _kernel[0] * _kernel[1] * self.batch_size
        # update (once a batch)
        write_energy = write_energy + self.get_energy_per_write(_in, _out_channel)
        read_energy = self.get_energy_per_read(_in, utils.ceil(_out_channel, const.phyArrColSize) * const.phyArrColSize)
        write_energy = write_energy + self.get_energy_per_write(_kernel[0] * _kernel[1] * _out_channel, utils.ceil(_in_channel, const.phyArrColSize) * const.phyArrColSize)

        return [read_energy, write_energy, mm_energy]

    def calc_fpConv(self, idx: int, layer: nn.Module) -> List[float]:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
        # choose a better allocation strategy
        _in = _kernel[0] * _kernel[1] * _in_channel
        if _bias == True:
            _in = _in + 1
        input_shape = self.shapes[idx]
        output_shape = self.shapes[idx + 1]
        unfold_L = output_shape[0] * output_shape[1]
        # forward
        write_energy = self.get_energy_per_write(input_shape[0] * input_shape[1], _in_channel) * self.batch_size
        mm_energy = self.get_energy_per_mm(_in, _out_channel) * unfold_L * self.batch_size 
        # backward
        mm_energy = mm_energy + self.get_energy_per_mm(_out_channel, _in) * input_shape[0] * input_shape[1] * self.batch_size
        mm_energy = mm_energy + self.get_energy_per_mm(input_shape[0] * input_shape[1], _in_channel) * _kernel[0] * _kernel[1] * self.batch_size
        # update (once a batch)
        write_energy = write_energy + self.get_energy_per_write(_in, _out_channel)
        read_energy = self.get_energy_per_read(_in, utils.ceil(_out_channel, const.phyArrColSize) * const.phyArrColSize)
        write_energy = write_energy + self.get_energy_per_write(_kernel[0] * _kernel[1] * _out_channel, utils.ceil(_in_channel, const.phyArrColSize) * const.phyArrColSize)

        return [read_energy, write_energy, mm_energy]

    def get_energy_per_write(self, m: int, n: int) -> float:
        # write all cells
        average_energy = self.average_write_energy[-1] / const.writeParallelism
        energy = average_energy * (m * n)

        return energy

    def get_energy_per_read(self, m: int, colSize: int) -> float:
        # all column per read
        voltage_square_mul_time = const.readV * const.readV * const.phyRdLatency
        cells = m * colSize

        energy = voltage_square_mul_time * (cells * self.average_conductance)
        energy = energy + m * const.readRowPeripheryEnergy + colSize * const.readColPeripheryEnergy

        return energy

    def get_energy_per_mm(self, row: int, col: int) -> float:
        average_p_square = 0.
        # TODO: what does the lateny means?
        voltage_square_mul_time = const.computeUnitV * const.computeUnitV * const.phyMMLatency
        for i in range(const.inVLevels):
            average_p_square = average_p_square + const.inVPD[i] * (i + 1) * (i + 1) / const.inVLevels / const.inVLevels
        average_energy = voltage_square_mul_time * average_p_square * self.average_conductance * const.inBits / const.inVBits
        mm_energy = (average_energy * const.phyArrColSize * const.phyArrRowSize) * row * col

        # circuit energy
        total_dac_energy = const.DACEnergy * row * const.inBits / const.inVBits
        total_adc_energy = const.ADCEnergy * col * const.inBits / const.inVBits
        periphery_energy = (
                                       const.computeRowPeripheryEnergy * row + const.computeColPeripheryEnergy * col) * const.inBits / const.inVBits
        
        self.adc_energy = self.adc_energy + total_adc_energy * self.batch_size
        self.dac_energy = self.dac_energy + total_dac_energy * self.batch_size
        self.calc_energy = self.calc_energy + (mm_energy + total_adc_energy + total_dac_energy + periphery_energy) * self.batch_size

        return mm_energy

    def get_average_write_energy(self) -> None:
        # per write : some cells in one row
        voltage_square_mul_time = const.writeV / 2 * const.writeV / 2 * const.phyWrLatency
        average_conductance = const.minConduct
        average_write_energy = (const.writeParallelism + 1) * [.0]
        for i in range(len(self.cell_pd)):
            average_conductance = average_conductance + self.cell_pd[i] * i * const.deltaConduct

        self.average_conductance = average_conductance
        
        # i cells in every writeParallelism cells need to write
        for i in range(1, const.writeParallelism + 1):
            # half selected row
            conductance = (const.phyArrColSize - i) * average_conductance
            # half selected col
            conductance = conductance + i * (const.phyArrRowSize - 1) * average_conductance
            # full selected cells
            conductance = conductance + i * 4 * average_conductance 
            # ref column need extra two cells
            if const.mode == 1:
                conductance = conductance + 2 * average_conductance
            # per write contains two ops : SET and RESET (* 2)
            average_write_energy[i] = voltage_square_mul_time * conductance * 2
        
        self.average_write_energy = average_write_energy
            
    def check(self) -> bool:
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

        if const.computeUseProbability is True:
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


class PE_latency():
    def __init__(self, in_data = 0, r_data = 0, inprecision = 8) -> None:
        
        self.buf_bitwidth = const.buf_bitwidth
        self.buf_cycle = const.buf_cycle
        self.buf_wlatency = utils.ceil(in_data * 8 / self.buf_bitwidth) * self.buf_cycle
        self.buf_rlatency = utils.ceil(r_data * 8 / self.buf_bitwidth) * self.buf_cycle
        
        self.pe_buf_write_latency = self.buf_wlatency # const.pe_buf_write_latency
        self.pe_buf_read_latency = self.buf_rlatency # const.pe_buf_read_latency

        self.read_row = const.read_row
        self.read_column = const.read_column

        self.dac_precision = const.dac_precision
        self.PE_group_DAC_num = const.PE_group_DAC_num
        self.group_num = const.group_num

        self.multiple_time = utils.ceil(inprecision / self.dac_precision) * utils.ceil(self.read_row / self.PE_group_DAC_num) * \
                             utils.ceil(self.read_column / self.PE_group_DAC_num)

        self.xbar_read_latency = const.xbar_read_latency
        self.xbar_latency = self.multiple_time * self.xbar_read_latency
        
        self.DAC_num = const.DAC_num
        self.ADC_num = const.ADC_num

        self.single_dac_latency = const.dac_latency
        self.dac_latency = self.multiple_time * self.single_dac_latency
        self.single_adc_latency = const.adc_latency
        self.adc_latency = self.multiple_time * self.single_adc_latency
        
        self.digital_period = const.digital_period
        self.iReg_latency = utils.ceil(self.read_row / self.PE_group_DAC_num) * utils.ceil(self.read_column / self.PE_group_DAC_num) * self.digital_period + \
                            self.multiple_time * self.digital_period
        self.shiftreg_latency = self.multiple_time * self.digital_period
        
        self.decoder_latency = const.decoder_latency
        self.input_demux_latency = self.multiple_time * self.decoder_latency
        self.adder_latency = utils.ceil(self.read_column / self.PE_group_DAC_num) * utils.ceil(math.log2(self.group_num)) * self.digital_period
        self.output_mux_latency = self.multiple_time * const.mux_latency
        
        self.computing_latency = self.dac_latency + self.xbar_latency + self.adc_latency
        self.oReg_latency = utils.ceil(self.read_column / self.PE_group_DAC_num) * self.digital_period
        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + \
            self.adder_latency + self.output_mux_latency + self.oReg_latency
        self.PE_latency = self.pe_buf_write_latency + self.pe_buf_read_latency + self.computing_latency + self.PE_digital_latency


class Tile_latency(PE_latency):
    def __init__(self, in_data = 0, r_data = 0, inprecision = 8, PE_num = 0) -> None:
        PE_latency.__init__(self, in_data = in_data, r_data = r_data, inprecision = inprecision)
        self.intra_tile_bandwidth = const.intra_tile_bandwidth
        self.merge_time = utils.ceil(math.log2(PE_num))
        self.tile_PE_num = const.tile_PE_num

        # 这段需要理解和修改
        if self.tile_PE_num[0] == 0:
            self.tile_PE_num[0] = 4
            self.tile_PE_num[1] = 4
        assert self.tile_PE_num[0] > 0, "PE number in one PE < 0"
        assert self.tile_PE_num[1] > 0, "PE number in one PE < 0"
        # total_level 中用到了
        self.tile_PE_total_num = self.tile_PE_num[0] * self.tile_PE_num[1]
        assert PE_num <= self.tile_PE_total_num, "PE number exceeds the range"

        self.read_column = const.read_column
        
        self.buf_wlatency = (self.PE.ADC_precision + self.merge_time) * self.read_column*PE_num/8

        self.total_level = utils.ceil(math.log2(self.tile_PE_total_num))

        self.jointmodule_latency = self.merge_time * self.digital_period
        self.transfer_latency = (self.total_level * (self.PE.ADC_precision + self.merge_time) - self.merge_time * (merge_time + 1) / 2) \
            * self.read_column / self.intra_tile_bandwidth
        self.tile_buf_rlatency = 0
        self.tile_buf_wlatency = self.buf_wlatency

        self.tile_latency = self.PE_latency + self.jointmodule_latency + self.transfer_latency + self.tile_buf_wlatency


class LatencyModule():
    def __init__(self, net: nn.Module, shapes: List, batch_size: int = 1) -> None:
        self.net = net
        self.shapes = shapes
        self.batch_size = batch_size

        self.write_latency = 0.
        self.mm_latency = 0.

        self.adc_latency = 0.
        self.dac_latency = 0.
        self.adder_latency = 0.
        
        self.total_latency = 0.
    
    def print_latency_info(self) -> None:
        self.calc_latency()
        print("Latency info:")
        print("     madel write latency : " % self.write_latency)
        print("     model mm latency : " % self.mm_latency)
        print("     adc latency : " % self.adc_latency)
        print("     dac latency : " % self.dac_latency)
        print("     adder latency : " % self.adder_latency)
        print("     total latency : " % self.total_latency)

    def get_info(self, net: nn.Module) -> None:
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Linear):
                res = self.calc_Linear_latency(layer)

    def calc_Linear_latency(self, layer: nn.Module) -> List[float]:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
        temp_tile_latency = Tile_latency(_in_channel, _out_channel, inprecision = 8, PE_num = 0)
        temp_tile_latency.outbuf.calculate_buf_read_latency(rdata=(self.graph.layer_tileinfo[layer_id]['max_column'] *
                                   outputbit * self.graph.layer_tileinfo[layer_id]['max_PE'] / 8))
        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
        merge_time = temp_tile_latency.tile_buf_rlatency + self.graph.inLayer_distance[0][layer_id] * \
                            (temp_tile_latency.digital_period + self.graph.layer_tileinfo[layer_id]['max_column'] *
                            self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
        # Todo: update merge time (adder tree) and transfer data volume
        transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                        output_size * outputbit / self.inter_tile_bandwidth)
        max_prelayer_time = 0
        temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
        for idx in temp_Inputindex:
            tmp_time = self.finish_time[layer_id+idx][-1]
            if tmp_time > max_prelayer_time:
                max_prelayer_time = tmp_time
        begin_time = max_prelayer_time
        compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
        return compute_time

    def calc_Convolution_latency(self, layer: nn.Module) -> List[float]:
        _in_channel, _out_channel, _kernel, _bias = layer.in_channels, layer.out_channels, layer.kernel_size, layer.hasBias
        temp_tile_latency = Tile_latency(_in_channel, _out_channel, inprecision = 8, PE_num = 0)
        compute_time_end = 0
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                if (i == 0) & (j == 0):
                    # the first output
                    indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                        kernelsize - padding, 0)) * inputbit / 8
                    # fill the line buffer
                    rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                    # from the line buffer to the input reg
                    temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                    '''和layer_id = 0 相比多出的部分,上'''
                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                    last_layer_finish_time = 0
                    # the finish time of all the required input data (in all input layers)
                    for idx in temp_Inputindex:
                        tmp_time = self.finish_time[layer_id + idx][-1]
                        if tmp_time > last_layer_finish_time:
                            last_layer_finish_time = tmp_time
                    '''和layer_id = 0 相比多出的部分,下'''
                    # layer_id = 0时begin_time=0, 并且compute_time不需要加上begin_time,
                    begin_time = last_layer_finish_time
                    compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + \
                                    begin_time
                    compute_time_end = max(compute_time_end, compute_time)
                    # consider the input data generation time
        return compute_time_end

    def cal_latency(self, arrX_size: int, arrY_size: int, type: int) -> None:
        if type == 0:
            num = min(arrX_size * arrY_size, const.phyArrayNum)
            sclar_da = (1.0 * const.phyArrayNum / const.dac_shared_ratio)
            sclar_ad = (1.0 * const.phyArrayNum / const.adc_shared_ratio)
            outI_latency = utils.ceil(num / sclar_da) * (const.dac_latency + const.phyMMLatency)
            adc_latency = utils.ceil(num / sclar_ad) * const.adc_latency
            add_all_latency = (num-1) * const.adder_latency
            return (outI_latency+adc_latency+add_all_latency) * const.inPluses
        elif type == 1:
            num = min(arrX_size * arrY_size, const.phyArrayNum)
            if num / const.parWrPhyNum+1 :
                num = num % const.parWrPhyNum
            else:
                num = num / const.parWrPhyNum
            return num * const.latencyWrSinglePhyArr
        elif type >= 3 and type <= 5 :
            num = min(arrX_size * arrY_size, const.phyArrayNum)
            sclar_da = (1.0 * const.phyArrayNum / const.dac_shared_ratio)
            sclar_ad = (1.0 * const.phyArrayNum / const.adc_shared_ratio)
            outI_latency = utils.ceil(num / sclar_da) * (const.dac_latency + const.phyMMLatency)
            adc_latency = utils.ceil(num / sclar_ad) * const.adc_latency
            add_all_latency = (num-1) * const.adder_latency
            if type == 3:
                return adc_latency * const.inPluses
            elif type == 4:
                return outI_latency * const.inPluses
            else:
                return add_all_latency * const.inPluses
        else:
            return 0
    
    def pipe_cal_latency(self, arrX_size: int, arrY_size: int, type: int) -> None:
        pass

def test():
    net = ConvMnist()
    manager = PerformanceManager(net, 28, 28)
    manager.print()

test()