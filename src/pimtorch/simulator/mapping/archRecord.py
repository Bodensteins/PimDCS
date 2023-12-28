import math
from functools import reduce
from src.pimtorch.config.globalCfg import globalCfg as cfg
from .archFunctional import segmentTree, Archid, ArchLevel, usageStatus


class Arch:
    def __init__(self, crx_n, pe_n, tile_n, bank_n, chip_n):

        is_log2_int = lambda x: math.log2(x).is_integer()
        # all input should be power of 2
        assert(is_log2_int(crx_n) and is_log2_int(pe_n) and is_log2_int(tile_n) and is_log2_int(bank_n)
               and is_log2_int(chip_n))

        self.chip_n = chip_n    # how many chips
        self.bank_n = bank_n    # how many bank per chip
        self.tile_n = tile_n    # tile per bank
        self.pe_n = pe_n        # pe per tile
        self.crx_n = crx_n      # crx per pe

        self.total_chip_n = self.chip_n
        self.total_bank_n = self.total_chip_n*self.bank_n
        self.total_tile_n = self.total_bank_n*self.tile_n
        self.total_pe_n = self.total_tile_n*self.pe_n
        self.total_crx_n = self.total_pe_n*self.crx_n

        self._crx_mask_size = int(math.log2(self.crx_n))
        self._pe_mask_size = int(math.log2(self.pe_n))
        self._tile_mask_size = int(math.log2(self.tile_n))
        self._bank_mask_size = int(math.log2(self.bank_n))
        self._chip_mask_size = int(math.log2(self.chip_n))

        self._mask_size_list = [0, self._crx_mask_size, self._pe_mask_size,
                                self._tile_mask_size, self._bank_mask_size, self._chip_mask_size]
        # The first 0 is placeholder

        for i in range(1, len(self._mask_size_list)):
            self._mask_size_list[i] += self._mask_size_list[i-1]

    # num == 1 --> get crx id in the pe
    # num == 2 --> get pe id in the tile, and others are similar
    def _get_local_id(self, pid, num = 1):
        mask = (1 << self._mask_size_list[num]) - 1
        local_id = (pid & mask) >> self._mask_size_list[num-1]
        return local_id

    def pid2Archid(self, pid) -> Archid:
        local_id_arr = list(map(self._get_local_id, [pid]*5, range(1, 6)))
        return Archid(*local_id_arr)

    def pid2LongLevelid(self, pid, level: ArchLevel):
        shift_bits = self._mask_size_list[level - 1]
        return (pid >> shift_bits) << shift_bits

    def pid2ShortLevelid(self, pid, level: ArchLevel):
        shift_bits = self._mask_size_list[level - 1]
        return pid >> shift_bits

    def Archid2pid(self, Arch_id: Archid) -> int:
        pid = reduce(lambda x, y: x+y, map(lambda x, y: x << y, Arch_id.get_arch_id_list(), self._mask_size_list[0:-1]))
        return pid

    def from_same_pe(self, pid1, pid2) -> bool:
        return self.is_same_level(pid1, pid2, ArchLevel.pe_level)

    def from_same_tile(self, pid1, pid2) -> bool:
        return self.is_same_level(pid1, pid2, ArchLevel.tile_level)

    def from_same_bank(self, pid1, pid2) -> bool:
        return self.is_same_level(pid1, pid2, ArchLevel.bank_level)

    def from_same_chip(self, pid1, pid2) -> bool:
        return self.is_same_level(pid1, pid2, ArchLevel.chip_level)

    def is_same_level(self, pid1, pid2, level: ArchLevel) -> bool:
        for lv in range(5, level-1, -1):
            if self._get_local_id(pid1, lv) != self._get_local_id(pid2, lv):
                return False
        return True

    def get_same_level_pid_list(self, pid, level: ArchLevel):
        mask = (1 << self._mask_size_list[level-1])-1
        stPid = pid - (pid & mask)
        edPid = stPid + (1 << self._mask_size_list[level-1])
        return range(stPid, edPid)

    def get_same_level_pid_slice(self, pid, level: ArchLevel):
        mask = (1 << self._mask_size_list[level-1])-1
        stPid = pid - (pid & mask)
        edPid = stPid + (1 << self._mask_size_list[level-1])
        return slice(stPid, edPid)

    def is_same_level_aid(self, aid1: Archid, aid2: Archid, level: ArchLevel) -> bool:
        return self.is_same_level(self.Archid2pid(aid1), self.Archid2pid(aid2), level)


class ArchCrxUsageRecord:
    def __init__(self, arch: Arch = None):
        if arch is None:
            self.arch = Arch(crx_n=cfg.crx_n, pe_n=cfg.pe_n, tile_n=cfg.tile_n,
                             bank_n=cfg.bank_n,
                             chip_n=cfg.chip_n)
        else:
            self.arch = arch
        self._crx_flag_map = segmentTree(self.arch.total_crx_n)

    def set_crx_used(self, pid):
        self._crx_flag_map.set(pid, pid, 1)  # [pid] = True

    def set_crx_free(self, pid):
        self._crx_flag_map.set(pid, pid, 0)
        # self._crx_flag_map[pid] = False

    # return pid list that is in the same level with pid
    def get_pid_list(self, pid, level: ArchLevel):
        return self.arch.get_same_level_pid_list(pid, level)

    def get_pid_slice(self, pid, level : ArchLevel):
        return self.arch.get_same_level_pid_slice(pid, level)

    def check_crx_used(self, pid):
        return self.check_crx_type(pid, ArchLevel.crx_level)

    def check_pe_used(self, pid):
        return self.check_crx_type(pid, ArchLevel.pe_level)

    def check_tile_used(self, pid):
        return self.check_crx_type(pid, ArchLevel.tile_level)

    def check_bank_used(self, pid):
        return self.check_crx_type(pid, ArchLevel.bank_level)

    def check_chip_used(self, pid):
        return self.check_crx_type(pid, ArchLevel.chip_level)

    def get_free_crx_pid_list(self, pid, level: ArchLevel):
        assert(level >= ArchLevel.crx_level)
        slices = self.arch.get_same_level_pid_slice(pid, level)
        step = self.arch.get_same_level_pid_slice(pid, ArchLevel.crx_level)
        step = step.stop-step.start
        pid_list = []
        for pid in range(slices.start, slices.stop, step):
            if self.check_crx_used(pid) == usageStatus.free:
                pid_list.append(pid)
        return pid_list

    # pid list 只包含每个free pe的第一个pid
    # 之后可以调用get_pid_list来得到具体一个Pe的所有pid，以下几个函数同理
    def get_free_pe_pid_list(self, pid, level: ArchLevel):
        assert(level >= ArchLevel.pe_level)
        slices = self.arch.get_same_level_pid_slice(pid, level)
        step = self.arch.get_same_level_pid_slice(pid, ArchLevel.pe_level)
        step = step.stop-step.start
        pid_list = []
        for pid in range(slices.start, slices.stop, step):
            if self.check_pe_used(pid)==usageStatus.free:
                pid_list.append(pid)
        return pid_list

    # pid list 只包含每个free tile 的第一个Pid
    def get_free_tile_pid_list(self, pid, level : ArchLevel):
        assert(level >= ArchLevel.tile_level)
        slices = self.arch.get_same_level_pid_slice(pid, level)
        step = self.arch.get_same_level_pid_slice(pid, ArchLevel.tile_level)
        step = step.stop-step.start
        pid_list = []
        for pid in range(slices.start, slices.stop, step):
            if self.check_tile_used(pid)==usageStatus.free:
                pid_list.append(pid)
        return pid_list

    def get_free_bank_pid_list(self, pid, level : ArchLevel):
        assert(level >= ArchLevel.bank_level)
        slices = self.arch.get_same_level_pid_slice(pid, level)
        step = self.arch.get_same_level_pid_slice(pid, ArchLevel.bank_level)
        step = step.stop-step.start
        pid_list = []
        for pid in range(slices.start, slices.stop, step):
            if self.check_bank_used(pid)==usageStatus.free:
                pid_list.append(pid)
        return pid_list

    def get_free_chip_pid_list(self, pid, level : ArchLevel):
        assert(level >= ArchLevel.chip_level)
        slices = self.arch.get_same_level_pid_slice(pid, level)
        step = self.arch.get_same_level_pid_slice(pid, ArchLevel.chip_level)
        step = step.stop-step.start
        pid_list = []
        for pid in range(slices.start, slices.stop, step):
            if self.check_chip_used(pid)==usageStatus.free:
                pid_list.append(pid)
        return pid_list

    def check_crx_type(self, pid, level: ArchLevel) -> usageStatus:
        slices = self.arch.get_same_level_pid_slice(pid, level)
        v = self._crx_flag_map.query(slices.start, slices.stop-1)
        if v == (slices.stop-slices.start):
            return usageStatus.full
        elif v == 0:
            return usageStatus.free
        else:
            return usageStatus.used

    def check_crx_free_num(self, pid, level : ArchLevel):
        slices = self.arch.get_same_level_pid_slice(pid, level)
        v = self._crx_flag_map.query(slices.start, slices.stop-1)
        return slices.stop-slices.start-v

    def current_pid_of_level(self, pid, level : ArchLevel):
        slices = self.arch.get_same_level_pid_slice(pid, level)
        return slices.start

    def next_pid_of_level(self, pid, level: ArchLevel):
        slices = self.arch.get_same_level_pid_slice(pid, level)
        return slices.stop

    def set_pid_slice_used(self, slices: slice):
        self._crx_flag_map.set(slices.start, slices.stop-1, 1)

    def set_pe_all_crx_used(self, pid):
        self.set_crx_type(pid, ArchLevel.pe_level, 1)

    def set_pe_all_crx_free(self, pid):
        self.set_crx_type(pid, ArchLevel.pe_level, 0)

    def set_tile_all_crx_used(self, pid):
        self.set_crx_type(pid, ArchLevel.tile_level, 1)

    def set_tile_all_crx_free(self, pid):
        self.set_crx_type(pid, ArchLevel.tile_level, 0)

    def set_bank_all_crx_used(self, pid):
        self.set_crx_type(pid, ArchLevel.bank_level, 1)

    def set_bank_all_crx_free(self, pid):
        self.set_crx_type(pid, ArchLevel.bank_level, 0)

    def set_chip_all_crx_used(self, pid):
        self.set_crx_type(pid, ArchLevel.chip_level, 1)

    def set_chip_all_crx_free(self, pid):
        self.set_crx_type(pid, ArchLevel.chip_level, 0)

    # 设置与pid 同一level层次的所有crx 为value状态 ,value为0或1, 0指free
    def set_crx_type(self, pid, level: ArchLevel, value):
        slices = self.arch.get_same_level_pid_slice(pid, level)
        self._crx_flag_map.set(slices.start, slices.stop-1, value)
