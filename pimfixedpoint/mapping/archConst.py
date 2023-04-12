from enum import Enum


class archConst:
    chip_n = 1
    bank_n = 8
    tile_n = 64
    pe_n = 16
    crx_n = 8

    arch_n_list = (crx_n, pe_n, tile_n, bank_n, chip_n)

    phyArraySize = (16, 64)

    splitBits = False
    cellBits = 1

    class PE:
        single_dac_area = 0.166015625
        single_adc_area = 1200
        single_SH_area = 0.0224609375
        single_cell_area = 0.004096
        adder_area = 0

        crx_n = 8
        DAC_n = 4
        ADC_n = 4
        adder_n = 1
        OR_n = 1
        IR_n = 1
        SH_n = 4
        SA_n = 1

    array_mode = 0  # 0:p&n 1:ref
    # times = 1 if array_mode == 1 else 2

    class PIMRunMode(Enum):
        # normal training, transient data used for backward is also stored in phy pim array.
        # refers to Pipelayer(hpca 2017), or Time (dac17) architecture.
        train = 0
        fast_mode_train = 1  # backend is digital
        inference = 2
        # transient data used for backward is stored in buffer.
        train_transientInBuffer = 3
