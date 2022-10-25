class energyInfo(object):
    memoryAccessEnergy = 0

    OffChipDramAccessEnergy = 0
    OffChipDramReadEnergy = 0
    OffChipDramWriteEnergy = 0

    ReRAMAccessEnergy = 0
    ReRAMReadEnergy = 0
    ReRAMWriteEnergy = 0

    BufferAccessEnergy = 0
    BufferReadEnergy = 0
    BufferWriteEnergy = 0

    # compute
    ComputeEnergy = 0
    ReRAMComputeEnergy = 0
    SampleAndHoldEnergy = 0
    ShiftAndAddEnergy = 0
    DACEnergy = 0
    ADCEnergy = 0

    BitWidthReductionEnergy = 0

    AdderEnergy = 0
    PartialSumAddEnergy = 0
    WeightUpdateAddEnergy = 0

    # other
    MaxPoolEnergy = 0
    ActivationEnergy = 0

    TransferEnergy = 0  # todo:needed?
    NetWorkOnChipEnergy = 0
    OffChipTransferEnergy = 0

    def __init__(self):
        super(energyInfo, self).__init__()