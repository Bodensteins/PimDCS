# from typing import List
#
#
# import sys
# sys.path.append("../mapping")
from mapping.archMapping import MapStrategyBase
from mapping.archConst import archConst
from mapping.archFunctional import ArchLevel


class AreaModule:
    def __init__(self, map_strategy: MapStrategyBase) -> None:
        """
        map_strategy: the strategy of mapping(include mapping record)
        """
        self.map_strategy = map_strategy

        # pe level
        self.array_area = archConst.phyArraySize[0] * archConst.phyArraySize[1] * \
                        archConst.PE.single_cell_area * archConst.PE.crx_n
        self.adc_area = archConst.PE.single_adc_area * archConst.PE.ADC_n
        self.dac_area = archConst.PE.single_dac_area * archConst.PE.DAC_n
        self.SH_area = archConst.PE.single_SH_area * archConst.PE.SH_n
        self.adder_area = archConst.PE.adder_area * archConst.PE.adder_n
        # TODO: other area

        self.pe_area = self.array_area + self.adc_area + self.dac_area + self.SH_area + self.adder_area

        # tile level
        # TODO: Peripheral circuits
        self.tile_area = self.pe_area * archConst.pe_n

        # bank level
        # TODO: Peripheral circuits
        self.bank_area = self.tile_area * archConst.tile_n

        # chip level
        # TODO: Peripheral circuits
        self.chip_area = self.bank_area * archConst.bank_n


    def get_pe_size(self) -> int:
        total_pe_n = self.map_strategy.archRecord.arch.total_pe_n
        free_pe_pidlist = self.map_strategy.archRecord.get_free_pe_pid_list(0, ArchLevel.chip_level)
        free_pe_n = len(free_pe_pidlist)

        print("total pe: %d" % (total_pe_n))
        print("free pe: %d" % (free_pe_n))
        return total_pe_n - free_pe_n

    def print(self) -> None:
        self.PE_size = self.get_pe_size()
        print("Area info(only consider pe level):")
        print("    pe_size : %d" % self.PE_size)
        print("    physical array area: %f mm^2" %
              (self.array_area * self.PE_size / 1e6))
        print("    dac area: %f mm^2" % (self.dac_area * self.PE_size / 1e6))
        print("    adc array area: %f mm^2" % (self.adc_area * self.PE_size / 1e6))
        print("    SH array area: %f mm^2" % (self.SH_area * self.PE_size / 1e6))
        print("    adder array area: %f mm^2" % (self.adder_area * self.PE_size / 1e6))
        print("    one pe area: %f mm^2" % (self.pe_area / 1e6))
        print("    total area: %f mm^2" % (self.pe_area * self.PE_size / 1e6))

    def printArch(self) -> None:
        print("Area info of entire Arch:")
        print("    total area of this chip is : %f mm^2" % (self.chip_area /1e6))
        print("    one chip has %d banks, area of each bank is  : %f mm^2" % (self.map_strategy.arch.bank_n, self.bank_area / 1e6))
        print("    one bank has %d tiles, area of each tile is  : %f mm^2" % (self.map_strategy.arch.tile_n, self.tile_area / 1e6))
        print("    one tile has %d pes  , area of each pe   is  : %f mm^2" % (self.map_strategy.arch.pe_n, self.pe_area / 1e6))