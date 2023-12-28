from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.addressParser import ArchLevel

class AreaModule:
    def __init__(self, resManager) -> None:
        """
        map_strategy: the strategy of mapping(include mapping record)
        """
        self.resManager = resManager

        # pe level
        self.array_area = cfg.crxShape[0] * cfg.crxShape[1] * cfg.singleCellArea * cfg.crx_n
        self.adc_area = cfg.singleADCArea * cfg.ADC_n
        self.dac_area = cfg.singleDACArea * cfg.DAC_n
        self.SH_area = cfg.singleSHArea * cfg.SH_n
        self.adder_area = cfg.adderArea * cfg.adder_n
        # TODO: other area

        self.pe_area = self.array_area + self.adc_area + self.dac_area + self.SH_area + self.adder_area

        # tile level
        # TODO: Peripheral circuits
        self.tile_area = self.pe_area * cfg.pe_n

        # bank level
        # TODO: Peripheral circuits
        self.bank_area = self.tile_area * cfg.tile_n

        # chip level
        # TODO: Peripheral circuits
        self.chip_area = self.bank_area * cfg.bank_n

    def get_pe_size(self) -> int:
        total_pe_n = self.resManager.getNum(archLevel=ArchLevel.pe)
        free_pe_n = self.resManager.getFreeNum(archLevel=ArchLevel.pe)

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
        print("    total area of this chip is : %f mm^2" % (self.chip_area / 1e6))
        print("    one chip has %d banks, area of each bank is  : %f mm^2" % (cfg.bank_n, self.bank_area / 1e6))
        print("    one bank has %d tiles, area of each tile is  : %f mm^2" % (cfg.tile_n, self.tile_area / 1e6))
        print("    one tile has %d pes  , area of each pe   is  : %f mm^2" % (cfg.pe_n, self.pe_area / 1e6))