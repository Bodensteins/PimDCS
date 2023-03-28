from enum import Enum
from math import floor

#############################  basic info ################################

writeV = 3.
readV = 1.
computeUnitV = 1.

phyArrRowSize = 128
phyArrColSize = 128
phyArrayNum = 8          # 1 PE contains 8 crossbars

inVBits = 1      # the input pulse (DAC) bits
inBits = 16      # the input for VMM bits
outBits = 8      # the output (ADC) bits
unitBits = 16    # weight bits
cellBits = 1     # the cell is 1 bit, so 8-bit weight need 8 cells to encode

cellsPerUnit = unitBits / cellBits
unitsPerPhyRow = phyArrColSize / cellsPerUnit

cellLevels = 1 << cellBits
cellsPerUnit = unitBits / cellBits
# there may be some cells unused in a row
# unitsPerPhyRow = phyArrColSize / cellsPerUnit
usedCellsPerPhyRow = unitsPerPhyRow * cellsPerUnit
inLevels = 1 << inBits
inVLevels = 1 << inVBits
outLevels = 1 << outBits
unitLevels = 1 << unitBits

minConduct = 1e-7
maxConduct = 1e-5
deltaConduct = (maxConduct - minConduct) / (cellLevels - 1)

dac_latency = 1
adc_latency = 6.25

mode = 0 #0:p&n 1:ref
times = 1 if mode == 1 else 2

class PIMRunMode(Enum):
    train = 0  #nomal training, transient data used for backward is also stored in phy pim array. refers to Pipelayer (hpca2017), or Time (dac17) architecture.
    fast_mode_train = 1 # backend is digital
    inference = 2
    train_transientInBuffer = 3 # transient data used for backward is stored in buffer.
runmode = PIMRunMode.train_transientInBuffer      # train_transientInBuffer
############################# area module ###############################

single_dac_area = 0.166015625
single_adc_area = 1200
single_SH_area = 0.0224609375
single_cell_area = 0.004096
adder_area = 0

dac_shared_ratio = 1    # 1 means every phy array has a set of dac/adc.  2 means, 2 phy array share 1 set of dac/adc and so on.
                        # 1 set of dac/adc means phyArray #rowsize DACs, and phyArray #colSize ADCs.
adc_shared_ratio = 128


########################### energy module(nJ) ###############################

readRowPeripheryEnergy = 0
readColPeripheryEnergy =0
readUseProbability = False #use probability to calculate or not
writeRowPeripheryEnergy = 0
writeColPeripheryEnergy = 0
writeParallelism = 128
writeUseProbability = False
DACPower = 0.00390625
ADCPower = 2.0
DACEnergy = dac_latency * DACPower * 1e-3 # nJ
ADCEnergy = adc_latency * ADCPower * 1e-3 # nJ
computeRowPeripheryEnergy = 0 #
computeColPeripheryEnergy = 0
adderEnergy = 0  #add vertical out
computeUseProbability = True
CellPD = [0.25, 0.25, 0.25, 0.25] #read probability distribution, if not illegal, exit
CellPDDefault = 0 #0:equal probabilty 1:100% max value
writePD = [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.3]
writePDDefault = 1
inVPD = [0., 1.]
inVPDDefault = 0 #0:equal probabilty 1:100% max value


########################### latency module(ns) ###############################

phyMMLatency = 10
phyRdLatency = 10
phyWrLatency = 50

# PE level
buf_bitwidth = 8
buf_cycle = 16

# 舍弃了
# pe_buf_write_latency = 20 //根据PE中的输入数据量indata得出
# pe_buf_read_latency = 2 //根据PE中从缓冲区到iReg的数据量rdata得出

default_inbuffer_size = 8 #// MNSIM 中是从Model_latency 逐级传到PE_latency
default_outbuffer_size = 8

read_row = 6
read_column = 6

dac_precision = 8
PE_group_DAC_num = 4
group_num = 2

dac_latency = 1
adc_latency = 6.25

digital_period = 4
# self.digital_period = 1/float(PEl_config.get('Digital module', 'Digital_Frequency'))*1e3
decoder_latency = 2
mux_latency = 3

DAC_num = 4
ADC_num = 1

# Tile level
intra_tile_bandwidth = 5
tile_PE_num = 16

# Operations that modify the delay time aren't locked.
class pim_latency(object):
    def __init__(self, s = 0, us = 0) -> None:
        self.run_latency_s = s
        self.run_latency_us = us
    
    def latency_add(self, time_ns) -> None:
        self.run_latency_us += time_ns / 1000.0
        add = floor(self.run_latency_us / 1e6)
        self.run_latency_s += add
        self.run_latency_us -= add * 1e6

    def latency_add_parallal(self, other) -> None:
        self.run_latency_s += other.run_latency_s
        self.run_latency_us += other.run_latency_us
    
    def print_latency(self) -> None:
        print("%d(s) %d(us)" % (self.run_latency_s, self.run_latency_us))