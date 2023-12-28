import numpy as np
from bidict import bidict
from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.resourceManager import ResourceManager
from src.pimtorch.simulator.addressParser import ArchLevel, ArchID

class MappingTable:
  def __init__(self, resourceManager: ResourceManager):
    # static metadata
    self.resourceManager = resourceManager
    self.logicalTensors = {}
    self.globalIdx2TensorID = np.zeros(shape=cfg.archInfo, dtype=int)
    self.globalIdx2mappingIdx = np.zeros(shape=(*cfg.archInfo, 2), dtype=int)
    self.name2TensorID = bidict()
    self.length = 0

  def addLogicalTensor(self, logicalTensor):
    tensorID = id(logicalTensor)
    self.logicalTensors[tensorID] = logicalTensor
    self.name2TensorID[logicalTensor.name] = tensorID
    self.length += logicalTensor.mapping2D.size
    for mappingIdx, levelID in np.ndenumerate(logicalTensor.mapping2D):
      npIdx = ArchID.archLevelID2archLevelIdx(levelID, logicalTensor.granularity)
      self.globalIdx2TensorID[npIdx] = tensorID
      self.globalIdx2mappingIdx[npIdx] = mappingIdx

  def removeLogicalTensor(self, logicalTensor):
    tensorID = id(logicalTensor)
    self.length -= logicalTensor.mapping2D.size
    for mappingIdx, levelID in np.ndenumerate(logicalTensor.mapping2D):
      npIdx = ArchID.archLevelID2archLevelIdx(levelID, logicalTensor.granularity)
      self.globalIdx2TensorID[npIdx] = 0
      self.globalIdx2mappingIdx[npIdx] = [-1, -1]
    self.logicalTensors.pop(tensorID)
    self.name2TensorID.pop(logicalTensor.name)

  def getLogicalTensor(self, archID: ArchID = None, archLevelIdx = None, tensorName = None):
    if archID is not None:
      tensorID = self.globalIdx2TensorID[archID.getArchLevelID()]
      if tensorID != 0:
        return self.logicalTensors[tensorID]
      else:
        return None
    elif archLevelIdx is not None:
      assert len(archLevelIdx) == 6
      tensorID = self.globalIdx2TensorID[archLevelIdx]
      if tensorID != 0:
        return self.logicalTensors[tensorID]
      else:
        return None
    elif tensorName is not None:
      return self.logicalTensors[self.name2TensorID[tensorName]]
    else:
      raise Exception("The function getLogicalTensor() expects the archID or the tensorName.")

  def getLogicalTensorName(self, tensorID):
    return self.name2TensorID.inverse[tensorID]


  def __len__(self):
    return self.length


