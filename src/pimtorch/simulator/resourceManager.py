import numpy as np
from functools import reduce
from enum import IntEnum
from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.addressParser import ArchLevel, ArchID

class UsageStatus(IntEnum):
  Free = 0  # all free
  Full = 1  # all used
  Used = 2  # partial used

class ResourceManager:
  def __init__(self):
    poolSize = reduce(lambda x, y: x * y, cfg.archInfo)
    self.globalIDPool = np.arange(poolSize, dtype=int).reshape(cfg.archInfo)
    self.usageBitMap = np.zeros_like(self.globalIDPool, dtype=int)

  def getUsageStatusByArchID(self, archID: ArchID):
    return self.getUsageStatusByArchLevelID(archID.getArchLevelID(ArchLevel.ou), ArchLevel.ou)

  def getUsageStatusByArchLevelID(self, archLevelID: int, archLevel: ArchLevel):
    status = []
    archLevelIdx = ArchID.archLevelID2archLevelIdx(archLevelID, archLevel)
    levelBitMap = self.usageBitMap
    for idx in archLevelIdx:
      levelBitMap = levelBitMap[idx]
      status.append(self.__checkUsageStatus__(levelBitMap))
    return status


  def getArchLevelIDs(self, parentID, parentLevel: ArchLevel, childrenLevel: ArchLevel):
    assert parentLevel <= childrenLevel
    archLevelIdx = ArchID.archLevelID2archLevelIdx(parentID, parentLevel)
    childrenID = self.globalIDPool[archLevelIdx]
    for level in range(ArchLevel.ou, childrenLevel, -1):
      childrenID = childrenID // cfg.archInfo[level - 1]
    return np.unique(childrenID)

  def getUsedArchLevelIDAndIdx(self, archLevel: ArchLevel):
    indices = np.nonzero(np.count_nonzero(self.usageBitMap, axis=tuple(range(archLevel.value - 6, 0))))
    idAndIdx = []
    for idx in zip(*indices):
      idAndIdx.append((ArchID.archLevelIdx2archLevelID(idx)[0], idx))
    return idAndIdx

  def getUsedNum(self, archLevel: ArchLevel = ArchLevel.ou):
    if archLevel == ArchLevel.ou:
      return np.count_nonzero(self.usageBitMap)
    else:
      return np.count_nonzero(self.usageBitMap.sum(axis=tuple(range(-(6 - archLevel), 0))))


  def getFreeNum(self, archLevel: ArchLevel = ArchLevel.ou):
    if archLevel == ArchLevel.ou:
      return np.count_nonzero(self.usageBitMap == 0)
    else:
      return np.count_nonzero(self.usageBitMap.sum(axis=tuple(range(-(6 - archLevel), 0))) == 0)

  def getNum(self, archLevel: ArchLevel = ArchLevel.ou):
    if archLevel == ArchLevel.ou:
      return self.usageBitMap.size
    else:
      n = 1
      for i in range(archLevel):
        n *= cfg.archInfo[i]
      return n

  def allocate(self, n = 1):
    if n > 1:
      archID = []
      for i in range(n):
        npIdx = self.__getFreeNpIdx__()
        self.__setUsageStatus__(npIdx, 1)
        archID.append(ArchID(self.globalIDPool[npIdx].item()))
    else:
      npIdx = self.__getFreeNpIdx__()
      self.__setUsageStatus__(npIdx, 1)
      archID = ArchID(self.globalIDPool[npIdx].item())
    return archID

  def free(self, archID):
    if isinstance(archID, list):
      for e in archID:
        self.__setUsageStatus__(e.getArchLevelIdx(), 0)
    elif isinstance(archID, ArchID):
      self.__setUsageStatus__(archID.getArchLevelIdx(), 0)
    else:
      raise TypeError("free() expects a parameter of ArchID list or ArchID.")

  def allocateArchLevel(self, archLevel: ArchLevel):
    targetNpIdx = None
    for archLevelID in np.ndindex(self.usageBitMap.shape[:archLevel.value]):
      if self.__checkUsageStatus__(self.usageBitMap[archLevelID]) == UsageStatus.Free:
        targetNpIdx = archLevelID
        break
    if targetNpIdx == None:
      raise RuntimeError("There are no free %s!" % archLevel.name)
    else:
      archID = self.globalIDPool[targetNpIdx].flatten().tolist()
      archID = [ArchID(globalID) for globalID in archID]
      self.__setUsageStatus__(targetNpIdx, 1)
      return archID


  def __getFreeNpIdx__(self):
    npIdx = np.where(self.usageBitMap == UsageStatus.Free)
    firstElement = []
    if npIdx[0].size == 0:
      raise RuntimeError("Out of Memory! (Total number of OU is %d)" % self.globalIDPool.size)
    else:
      for level in npIdx:
        assert level.size > 0
        firstElement.append(np.array(level[0]))
    return tuple(firstElement)

  def __setUsageStatus__(self, npIdx, status: int):
    assert status == 0 or status == 1
    self.usageBitMap[npIdx] = status

  def __checkUsageStatus__(self, bitMap):
    if np.all(bitMap == UsageStatus.Free):
      return UsageStatus.Free
    elif np.all(bitMap == UsageStatus.Full):
      return UsageStatus.Full
    else:
      return UsageStatus.Used

  def __len__(self):
    return self.usageBitMap.size
