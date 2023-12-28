from enum import IntEnum
from src.pimtorch.config.globalCfg import globalCfg as cfg

class ArchLevel(IntEnum):
  offchip = 0
  chip = 1
  bank = 2
  tile = 3
  pe = 4
  crx = 5
  ou = 6


class ArchID:
  def __init__(self, globalID: int):
    self.globalID = globalID
    for i in range(6, 0, -1):
      arch_name = ArchLevel(i).name
      self.__setattr__(arch_name + "ID", globalID)
      self.__setattr__(arch_name, globalID % cfg.archInfo[i - 1])
      globalID = globalID // cfg.archInfo[i - 1]

  def getGlobalID(self):
    return self.globalID

  def getArchLevelID(self, archLevel: ArchLevel = None):
    if archLevel is None:
      return (self.chip, self.bank, self.tile, self.pe, self.crx, self.ou)
    elif archLevel == ArchLevel.offchip:
      raise ValueError("The off-chip level has no arch level id.")
    return self.__getattribute__(archLevel.name + "ID")

  def getArchLevelIdx(self, archLevel: ArchLevel = None):
    if archLevel is None:
      return (self.chip, self.bank, self.tile, self.pe, self.crx, self.ou)
    else:
      return self.__getattribute__(archLevel.name)

  def getCommonArchLevelID(self, archID):
    lArchLevelID = self.getArchLevelID()
    rArchLevelID = archID.getArchLevelID()
    for i, (l, r) in enumerate(zip(lArchLevelID[::-1], rArchLevelID[::-1])):
      if l == r:
        return (l, ArchLevel(6 - i))

  def __str__(self):
    return ("[chip, bank, tile, pe, crx, ou] = [{}, {}, {}, {}, {}, {}]\n"
            "[chipID, bankID, tileID, peID, crxID, ouID] = [{}, {}, {}, {}, {}, {}]"
            .format(self.chip, self.bank, self.tile, self.pe, self.crx, self.ou,
                    self.chipID, self.bankID, self.tileID, self.peID, self.crxID, self.ouID))

  @classmethod
  def archLevelID2archLevelIdx(cls, archLevelID: int, archLevel):
    idx = []
    if isinstance(archLevel, ArchLevel):
      archLevel = archLevel.value
    for i in range(archLevel, 0, -1):
      idx.append(archLevelID % cfg.archInfo[i - 1])
      archLevelID = archLevelID // cfg.archInfo[i - 1]
    idx.reverse()
    return tuple(idx)

  @classmethod
  def archLevelIdx2archLevelID(cls, archLevelIdx):
    archLevelID = archLevelIdx[0]
    archLevel = ArchLevel(len(archLevelIdx))
    for level in range(1, len(archLevelIdx)):
      archLevelID = archLevelID * getattr(cfg, ArchLevel(level+1).name + "_n") + archLevelIdx[level]
    return (archLevelID, archLevel)
