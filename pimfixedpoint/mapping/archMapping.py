import math
import numpy as np
# from functools import reduce
from mapping.archConst import archConst
from mapping.archRecord import ArchCrxUsageRecord, Arch
from mapping.archFunctional import ArchLevel, usageStatus


class LogicalArray:
    def __init__(self, unique_key, logicalArraySize: tuple, physicalArraySize: tuple, splitBits: bool,
                 bitWidth: int, cellBits: int):
        self.unique_key = unique_key
        self.logicalArraySize = logicalArraySize
        self.physicalArraySize = physicalArraySize
        self.bitWidth = bitWidth
        self.splitBits = splitBits

        r = math.ceil(logicalArraySize[0] / physicalArraySize[0])
        cellNum = math.ceil(bitWidth / cellBits)
        if self.splitBits:
            c = math.ceil(logicalArraySize[1] / physicalArraySize[1]) * cellNum
        else:
            c = math.ceil(logicalArraySize[1]/(physicalArraySize[1]//cellNum))
        self.pidMap2D = np.full([r, c], np.nan, dtype='int64')
        self.needPhyArrayN = r*c
    
    def __str__(self):
        return self.pidMap2D.__str__()


class MapStrategyBase:
    def __init__(self, arch=None):
        if arch is None:
            self.arch = Arch(*archConst.arch_n_list)
        else:
            self.arch = arch
        self.archRecord = ArchCrxUsageRecord(arch)
        self.logicalArrayDict = {}

    def allocLogicalArray(self, *args):
        pass

    def getLogicalArrayMap(self, *args):
        pass

    def printByLevel(self, pid: int, archLevel: ArchLevel) -> None:
        """
        given pid, print the status of all arrays by ArchLevel.
        because you can easily check one xbar by "check_crx_used()", so this func doesn't support ArchLevel.crx_level.
        """
        assert(archLevel > ArchLevel.crx_level)
        pids = self.arch.get_same_level_pid_list(pid, archLevel)
        assert(len(pids) > 0)
        pid_start, pid_end = pids[0], pids[-1]
        step = 1 << self.arch._mask_size_list[archLevel - 2]
        pid_t = pid_start
        pidList = []
        while pid_t <= pid_end:
            pidList.append(pid_t)
            pid_t += step
        
        strs = ["     ", " Xbar ", "  PE  ", " Tile ", " Bank ", " Chip "]
        length = len(pidList) * 4
        archid = self.arch.pid2Archid(pid)
        print("Pid {} belongs to arch [Xbar:{}, PE:{}, Tile:{}, Bank:{}, Chip:{}].".format(pid, *archid.get_arch_id_list()))
        print("You are checking {} status of the same {} with the pid.".format(strs[archLevel - 1], strs[archLevel]))
        id = archid.get_arch_id_list()[archLevel - 1]
        print("-" * length + strs[archLevel] + str(id) + "-" * length)
        line = "|" + strs[archLevel - 1] + "|"
        for i in range(len(pidList)):
            line += (5 - len(str(i))) * " " + str(i) + "  |"
        print(line)
        line = "|usage |"
        for pid in pidList:
            status = self.archRecord.check_crx_type(pid, archLevel - 1)
            if status == usageStatus.free:
                line += "  Free |"
            elif status == usageStatus.full:
                line += "  Full |"
            else:
                line += "  used |"
        print(line)
        
    def printByLayer(self, unique_key) -> None:
        keys = self.logicalArrayDict.keys()
        assert (unique_key in keys), "unique_key not founded in pidMap2D"

        logicArray = self.logicalArrayDict[unique_key]
        pids = logicArray.pidMap2D

        for pid in np.nditer(pids):
            aid = self.arch.pid2Archid(pid)
            # TODO: output to file instead of terminal
            print(aid)


# 默认的mapping策略十分简单
# 只保证一个pe里面都是来自同一个logicalArray的数据
# 因此，它没有保证一个logicalArray数据一定在同一tile的pe等等
class DefaultMapStrategy(MapStrategyBase):
    def __init__(self, arch: Arch = None):
        super().__init__(arch)
        self.current_pid = 0
    
    def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
        # assert unique_key not in self.logicalArrayDict
        if unique_key in self.logicalArrayDict:
            raise Exception(unique_key + "is not an unique key, it has been added to logic array dict.")

        now = self.logicalArrayDict[unique_key] = LogicalArray(unique_key, logicalArraySize, archConst.phyArraySize,
                                                               archConst.splitBits, bitWidth, archConst.cellBits)

        # 保证PE的数据来自同一logical Array
        if self.archRecord.check_pe_used(self.current_pid) != usageStatus.free:
            self.current_pid = self.archRecord.next_pid_of_level(self.current_pid, ArchLevel.pe_level)

        pid_list = self.allocPhysicalArray(now.needPhyArrayN)

        for i, (row, col) in enumerate(np.ndindex(*now.pidMap2D.shape)):
            pid = pid_list[i]
            now.pidMap2D[row, col] = pid
    
    def allocPhysicalArray(self, number):
        pid_list = range(self.current_pid, self.current_pid+number)
        self.archRecord.set_pid_slice_used(slice(self.current_pid, self.current_pid + number))
        return pid_list

    def getLogicalArrayMap(self, unique_key):
        return self.logicalArrayDict[unique_key]


if __name__ == "__main__":
    ac = Arch(4, 16, 8, 1, 128)
    pid = int("11000100001", 2)
    print(pid)
    aid = ac.pid2Archid(pid)
    print(aid)
    print(ac.Archid2pid(aid))

    print(ac.is_same_level(64, 65, ArchLevel.tile_level))
    
    dmap = DefaultMapStrategy(ac)
    dmap.current_pid = 99
    dmap.allocLogicalArray(5, (9, 9), 8)
    temp = dmap.getLogicalArrayMap(5)
    print(temp)
    dmap.printByLevel(99, archLevel=ArchLevel.pe_level)
    print(dmap.archRecord.get_free_pe_pid_list(64, ArchLevel.bank_level))

    # tree = segmentTree(16)
    # num = int(input())
    # for i in range(num):
    #     l, r, t = map(int, input().split())
    #     tree.set(l, r, t)
    #     print(tree.query(0, 15))


    

    
