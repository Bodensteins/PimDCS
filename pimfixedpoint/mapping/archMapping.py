import math
import torch
import numpy as np
import uuid
from enum import Enum
from .archConst import archConst
from .archRecord import ArchCrxUsageRecord, Arch
from .archFunctional import ArchLevel, usageStatus


class FillingType(Enum):
  FixedSize = "fixed-size"
  Multiple = "multiple"
  All = "all"

  def __str__(self):
    return self.value

class LogicalArray:
  def __init__(self, unique_key, logicalArraySize: tuple, physicalArraySize: tuple, splitBits: bool,
               bitWidth: int, cellBits: int, device: torch.device):
    """
    The logical array is a virtual reference of a matrix. If the size of logical array is larger than
    the size of physical array, the logical array would be divided into multiple sub-blocks which will be
    stored into multiple physical array.
    @param unique_key: The unique key of logical array.
    @param logicalArraySize: The size of logical array.
    @param physicalArraySize: The size of physical array.
    @param splitBits: Splits data by bits and distributes them to different physical arrays.
    @param bitWidth: Bits width of data.
    @param cellBits: Cell bits.
    """
    self.unique_key = unique_key
    self.logicalArraySize = logicalArraySize
    self.physicalArraySize = physicalArraySize
    self.bitWidth = bitWidth
    self.splitBits = splitBits
    self.cellBits = cellBits
    self.cellNumPerValue = math.ceil(bitWidth / cellBits)
    self.device = device
    self.intervalDiffCntList = [torch.zeros(self.logicalArraySize, dtype=torch.int64, device=self.device) for i in
                                range(self.cellNumPerValue)]

    r = math.ceil(logicalArraySize[0] / physicalArraySize[0])
    if self.splitBits:
      c = math.ceil(logicalArraySize[1] / physicalArraySize[1]) * self.cellNumPerValue
    else:
      c = math.ceil(logicalArraySize[1] * bitWidth / (physicalArraySize[1] * cellBits))
    self.pidMap2D = np.full([r, c], -1, dtype=np.int64)

  def updateDiffCntList(self, diff):
    for i in range(self.cellNumPerValue):
      for j in range(self.cellBits):
        self.intervalDiffCntList[i] += diff & 1
        diff.bitwise_right_shift_(1)

  def resetDiffCntList(self, wlManager):
    # flush intervalDiffCntList
    if self.splitBits:
      colOffset = math.ceil(self.logicalArraySize[1] / self.physicalArraySize[1])
      for i in range(self.cellNumPerValue):
        row_indices = [] if self.physicalArraySize[0] > self.intervalDiffCntList[i].shape[0] else torch.arange(
          self.physicalArraySize[0], self.intervalDiffCntList[i].shape[0], self.physicalArraySize[0])
        for j, row_split in enumerate(torch.tensor_split(self.intervalDiffCntList[i], row_indices, dim=0)):
          col_indices = [] if self.physicalArraySize[1] > row_split.shape[1] else torch.arange(
            self.physicalArraySize[1], row_split.shape[1], self.physicalArraySize[1])
          for k, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
            col_idx = i * colOffset + k
            pid = self.pidMap2D[j][col_idx]
            wlManager.updateWriteCountDict(pid, col_split)
    else:
      countMap = torch.dstack(tuple(self.intervalDiffCntList)).view(
        [self.logicalArraySize[0], self.cellNumPerValue * self.logicalArraySize[1]])
      colChunkSize = math.ceil(countMap.shape[1] / self.pidMap2D.shape[1])
      row_indices = [] if self.physicalArraySize[0] > countMap.shape[0] else torch.arange(self.physicalArraySize[0],
                                                                                          countMap.shape[0],
                                                                                          self.physicalArraySize[0])
      for i, row_split in enumerate(torch.tensor_split(countMap, row_indices, dim=0)):
        col_indices = [] if colChunkSize > row_split.shape[1] else torch.arange(colChunkSize, row_split.shape[1],
                                                                                colChunkSize)
        for j, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
          pid = self.pidMap2D[i][j]
          wlManager.updateWriteCountDict(pid, col_split)

    # reset intervalDiffCntList
    self.intervalDiffCntList = list(map(lambda x: x.zero_(), self.intervalDiffCntList))

  def __str__(self):
    return self.pidMap2D.__str__()


class MapStrategyBase:
  def __init__(self, arch=None, splitBits=False, device=torch.device("cpu")):
    if arch is None:
      self.arch = Arch(*archConst.arch_n_list)
    else:
      self.arch = arch
    self.device = device
    self.archRecord = ArchCrxUsageRecord(arch)

    # wear-leveling metadata
    self.splitBits = splitBits
    self.cellCurrentTensorDict = {}
    self.cellWriteCountDict = {}
    self.logicalArrayDict = {}
    self.pid2lid = {}
    self.peid2pid = {}
    # physicalArrayInfo: ("pid", "pe_id", "iwc", "twc")
    self.physicalArrayInfo = np.empty((0, 4), dtype=np.uint64)
    # peInfo: ("pe_id", "pe_iwc", "pe_twc")
    self.peInfo = np.empty((0, 3), dtype=np.uint64)
    self.peTileBankID = {}
    self.pumaPETileBankID = {}    # pimtorch pe_id to puma pCore and pTile

  def allocLogicalArray(self, *args, **kwargs):
    pass

  def getLogicalArrayMap(self, unique_key, *args, **kwargs):
    return self.logicalArrayDict[unique_key]


  def printByLevel(self, pid: int, archLevel: ArchLevel) -> None:
    """
    given pid, print the status of all arrays by ArchLevel.
    because you can easily check one xbar by "check_crx_used()", so this func doesn't support ArchLevel.crx_level.
    """
    assert (archLevel > ArchLevel.crx_level)
    pids = self.arch.get_same_level_pid_list(pid, archLevel)
    assert (len(pids) > 0)
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
  def __init__(self, arch: Arch = None, splitBits=False, device=torch.device("cpu")):
    super().__init__(arch, splitBits, device)
    self.current_pid = 0

  def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
    # assert unique_key not in self.logicalArrayDict
    if unique_key in self.logicalArrayDict:
      raise Exception(unique_key + "is not an unique key, it has been added to logic array dict.")

    self.logicalArrayDict[unique_key] = LogicalArray(unique_key, (logicalArraySize[1], logicalArraySize[0]),
                                                     archConst.phyArraySize, self.splitBits, bitWidth, archConst.cellBits, self.device)

    # 保证PE的数据来自同一logical Array
    while self.archRecord.check_pe_used(self.current_pid) != usageStatus.free:
      self.current_pid = self.archRecord.next_pid_of_level(self.current_pid, ArchLevel.pe_level)

    pid_list = self.allocPhysicalArray(self.logicalArrayDict[unique_key].pidMap2D.size)

    for i, (row, col) in enumerate(np.ndindex(self.logicalArrayDict[unique_key].pidMap2D.shape)):
      pid = pid_list[i]
      # pe_id = self.arch.pid2GlobalPeid(pid)
      pe_id = self.arch.pid2ShortLevelid(pid, ArchLevel.pe_level)
      self.logicalArrayDict[unique_key].pidMap2D[row, col] = pid
      self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
      self.pid2lid[pid] = unique_key

  def allocPhysicalArray(self, number):
    pid_list = range(self.current_pid, self.current_pid + number)
    self.archRecord.set_pid_slice_used(slice(self.current_pid, self.current_pid + number))
    for pid in pid_list:
      # pe_id = self.arch.pid2GlobalPeid(pid)
      pe_id = self.arch.pid2ShortLevelid(pid, ArchLevel.pe_level)
      if pe_id not in self.peid2pid:
        self.peid2pid[pe_id] = [pid]
        self.peInfo = np.append(self.peInfo, np.array([[pe_id, 0, 0]], dtype=np.uint64), axis=0)
      else:
        self.peid2pid[pe_id].append(pid)
      self.cellWriteCountDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
      self.cellCurrentTensorDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
    self.current_pid += number
    return pid_list


class SaveSpaceMapStrategy(MapStrategyBase):
  def __init__(self, arch: Arch = None, splitBits=False, device=torch.device("cpu")):
    """
    SaveSpaceMapStrategy prefers to find free Crossbar for allocation and does not guarantee that the data
    within PE belong to the same logical matrix.
    @param arch: architecture config
    """
    super().__init__(arch, splitBits, device)
    self.current_pid = 0

  def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
    if unique_key in self.logicalArrayDict:
      raise Exception(unique_key + "is not an unique key, it has been added to logic array dict.")

    self.logicalArrayDict[unique_key] = LogicalArray(unique_key, (logicalArraySize[1], logicalArraySize[0]),
                                                     archConst.phyArraySize, self.splitBits, bitWidth, archConst.cellBits, self.device)
    for row, col in np.ndindex(self.logicalArrayDict[unique_key].pidMap2D.shape):
      pid = self.allocPhysicalArray()
      # pe_id = self.arch.pid2GlobalPeid(pid)
      pe_id = self.arch.pid2ShortLevelid(pid, ArchLevel.pe_level)
      self.logicalArrayDict[unique_key].pidMap2D[row, col] = pid
      self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
      self.pid2lid[pid] = unique_key

  def allocPhysicalArray(self):
    pid = self.archRecord.get_free_crx_pid_list(self.current_pid, ArchLevel.crx_level)[0]
    # pe_id = self.arch.pid2GlobalPeid(pid)
    pe_id = self.arch.pid2ShortLevelid(pid, ArchLevel.pe_level)
    if pe_id not in self.peid2pid:
      self.peid2pid[pe_id] = [pid]
      self.peInfo = np.append(self.peInfo, np.array([[pe_id, 0, 0]], dtype=np.uint64), axis=0)
    else:
      self.peid2pid[pe_id].append(pid)
    self.cellWriteCountDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
    self.cellCurrentTensorDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
    self.archRecord.set_crx_used(pid)
    self.current_pid += 1
    return pid

  def fill_final_used_pe(self):
    pe_id = self.peInfo[-1][0]
    if self.archRecord.check_pe_used(self.peid2pid[pe_id][0]) != usageStatus.full:
      pid_list = self.archRecord.get_free_crx_pid_list(self.peid2pid[pe_id][0], ArchLevel.pe_level)
      self.allocLogicalArray(unique_key="Placeholder_" + str(uuid.uuid4()), logicalArraySize=(1, len(pid_list)),
                             bitWidth=archConst.phyArraySize[1])

  def fill_to(self, mode: FillingType, value, wlConfig):
    if mode == FillingType.FixedSize:
      pid_end = math.ceil((value * 1024 * 1024 * 1024 * 8) / (archConst.phyArraySize[0] * archConst.phyArraySize[1]))
    elif mode == FillingType.Multiple:
      pid_end = int(self.current_pid * value)
    elif mode == FillingType.All:
      pid_end = self.arch.total_crx_n
    else:
      raise ValueError("Unsupported fill mode: %s. Please use FixedSize or Multiple mode." % mode)
    if pid_end > self.current_pid:
      logicalArraySize = (1, pid_end - self.current_pid)
      self.allocLogicalArray(unique_key="Placeholder-" + str(uuid.uuid4()), logicalArraySize=logicalArraySize,
                             bitWidth=archConst.phyArraySize[1])
    self.fill_final_used_pe()
    # wlConfig.topKSize = int(self.peInfo.shape[0] / 2 * wlConfig.topK * 0.01)
    wlConfig.topKSize = int(self.peInfo.shape[0] * wlConfig.topK * 0.01)
    for pe_id in self.peid2pid:
      self.peTileBankID[pe_id] = (self.arch.pid2LongLevelid(self.peid2pid[pe_id][0], ArchLevel.tile_level),
                                  self.arch.pid2LongLevelid(self.peid2pid[pe_id][0], ArchLevel.bank_level))
      # puma arch: 16 xbar -> 1 vMVMU, 6 pMVMU -> 1 pCore, 8 pCore -> 1 pTile
      pMVMU = pe_id % 6
      pCore = pe_id // 6
      pTile = pe_id // 48 + 2
      self.pumaPETileBankID[pe_id] = (pMVMU, pCore, pTile)


class AlignMapStrategy(MapStrategyBase):
  def __init__(self, arch: Arch = None, splitBits=False, device=torch.device("cpu")):
    super().__init__(arch, splitBits, device)
    self.current_pid = 0
    self.placeholder_pid = []
    self.placeholder_unique_key = "Placeholder-" + str(uuid.uuid4())

  def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
    if unique_key in self.logicalArrayDict:
      raise Exception(unique_key + "is not an unique key, it has been added to logic array dict.")
    self.logicalArrayDict[unique_key] = LogicalArray(unique_key, (logicalArraySize[1], logicalArraySize[0]),
                                                     archConst.phyArraySize, self.splitBits, bitWidth, archConst.cellBits, self.device)
    for row in self.logicalArrayDict[unique_key].pidMap2D:
      for col_idx in range(0, len(row), self.arch.crx_n):
        allocNum = min(len(row) - col_idx, self.arch.crx_n)
        pid_list = self.allocPE(unique_key, allocNum)
        row[col_idx: col_idx + allocNum] = pid_list


  def allocPE(self, unique_key, physicalArrayUsedNum):
    pid_list = []
    for i in range(physicalArrayUsedNum):
      pid = self.allocPhysicalArray()
      pid_list.append(pid)
      self.pid2lid[pid] = unique_key
    for i in range(physicalArrayUsedNum, self.arch.crx_n):
      pid = self.allocPhysicalArray()
      self.placeholder_pid.append(pid)
      self.pid2lid[pid] = self.placeholder_unique_key
    return torch.tensor(pid_list, dtype=torch.int64)

  def allocPhysicalArray(self):
    pid = self.archRecord.get_free_crx_pid_list(self.current_pid, ArchLevel.crx_level)[0]
    pe_id = self.arch.pid2ShortLevelid(pid, ArchLevel.pe_level)
    if pe_id not in self.peid2pid:
      self.peid2pid[pe_id] = [pid]
      self.peInfo = np.append(self.peInfo, np.array([[pe_id, 0, 0]], dtype=np.uint64), axis=0)
    else:
      self.peid2pid[pe_id].append(pid)
    self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
    self.cellWriteCountDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
    self.cellCurrentTensorDict[pid] = torch.zeros(archConst.phyArraySize, dtype=torch.int64, device=self.device)
    self.archRecord.set_crx_used(pid)
    self.current_pid += 1
    return pid

  def fill_to(self, mode: FillingType, value, wlConfig):
    if mode == FillingType.FixedSize:
      # value n means fill to n GB
      pid_end = math.ceil((value * 1024 * 1024 * 1024 * 8) / (archConst.phyArraySize[0] * archConst.phyArraySize[1]))
    elif mode == FillingType.Multiple:
      pid_end = int(self.current_pid * value)
    elif mode == FillingType.All:
      pid_end = self.arch.total_crx_n
    else:
      raise ValueError("Unsupported fill mode: %s. Please use FixedSize or Multiple mode." % mode)
    while (self.current_pid < pid_end):
      pid = self.allocPhysicalArray()
      self.placeholder_pid.append(pid)
      self.pid2lid[pid] = self.placeholder_unique_key
    self.allocPlaceHolder()
    # wlConfig.topKSize = int(self.peInfo.shape[0] / 2 * wlConfig.topK * 0.01)
    wlConfig.topKSize = int(self.peInfo.shape[0] * wlConfig.topK * 0.01)
    for pe_id in self.peid2pid:
      self.peTileBankID[pe_id] = (self.arch.pid2LongLevelid(self.peid2pid[pe_id][0], ArchLevel.tile_level),
                                  self.arch.pid2LongLevelid(self.peid2pid[pe_id][0], ArchLevel.bank_level))
      # puma arch: 16 xbar -> 1 vMVMU, 6 pMVMU -> 1 pCore, 8 pCore -> 1 pTile
      pMVMU = pe_id % 6
      pCore = pe_id // 6
      pTile = pe_id // 48 + 2
      self.pumaPETileBankID[pe_id] = (pMVMU, pCore, pTile)



  def allocPlaceHolder(self):
    if len(self.placeholder_pid) > 0:
      logicalArraySize = (1, len(self.placeholder_pid))
      self.logicalArrayDict[self.placeholder_unique_key] = LogicalArray(self.placeholder_unique_key, logicalArraySize,
                                                                        archConst.phyArraySize, self.splitBits,
                                                                        archConst.phyArraySize[1], archConst.cellBits,
                                                                        self.device)
      self.logicalArrayDict[self.placeholder_unique_key].pidMap2D = np.array(self.placeholder_pid, dtype=np.int64).reshape(logicalArraySize)



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
