from enum import Enum


############################# area module ###############################
phyArrRowSize = 128
phyArrColSize = 128
phyArrayNum = 8          # 1 PE contains 8 crossbars

unitBits = 16    # weight bits
cellBits = 1     # the cell is 1 bit, so 8-bit weight need 8 cells to encode

cellsPerUnit = unitBits / cellBits
unitsPerPhyRow = phyArrColSize / cellsPerUnit

single_dac_area = 0.166015625
single_adc_area = 1200
single_SH_area = 0.0224609375
single_cell_area = 0.004096
adder_area = 0

dac_shared_ratio = 1    # 1 means every phy array has a set of dac/adc.  2 means, 2 phy array share 1 set of dac/adc and so on.
                        # 1 set of dac/adc means phyArray #rowsize DACs, and phyArray #colSize ADCs.
adc_shared_ratio = 128

mode = 0 #0:p&n 1:ref
times = 1 if mode == 1 else 2

class PIMRunMode(Enum):
    train = 0  #nomal training, transient data used for backward is also stored in phy pim array. refers to Pipelayer (hpca2017), or Time (dac17) architecture.
    fast_mode_train = 1 # backend is digital
    inference = 2
    train_transientInBuffer = 3 # transient data used for backward is stored in buffer.
runmode = PIMRunMode.train_transientInBuffer      # train_transientInBuffer


########################### energy module(nJ) ###############################
readRowPeripheryEnergy = 0
readColPeripheryEnergy =0
readUseProbability = False #use probability to calculate or not
writeRowPeripheryEnergy = 0
writeColPeripheryEnergy = 0
writeParallelism = 128
writeUseProbability = True
DACPower = 0.00390625
ADCPower = 2
computeRowPeripheryEnergy = 0 #
computeColPeripheryEnergy = 0
adderEnergy = 0  #add vertical out
computeUseProbability = True
CellPD = [0.25, 0.25, 0.25, 0.25] #read probability distribution, if not illegal, exit
CellPDDefault = 0 #0:equal probabilty 1:100% max value
writePD = [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.3]
writePDDefault = 1
inVPD = [0.5, 0.5]
inVPDDefault = 0 #0:equal probabilty 1:100% max value


########################### latency module ###############################