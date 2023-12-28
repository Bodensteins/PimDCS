import numpy as np
from src.pimtorch.simulator.resourceManager import ResourceManager
from src.pimtorch.simulator.addressParser import ArchLevel

class Allocator:
  @staticmethod
  def allocateLogicalTensor(mapping2D: np.ndarray, resourceManager: ResourceManager, **kwargs):
    with np.nditer(mapping2D, op_flags=['readwrite']) as it:
      for x in it:
        archID = resourceManager.allocate()
        x[...] = archID.getGlobalID()


class AlignAllocator(Allocator):
  @staticmethod
  def allocateLogicalTensor(mapping2D: np.ndarray, resourceManager: ResourceManager, **kwargs):
    granularity = kwargs["granularity"]
    for row in mapping2D:
      archIDs = []
      while len(archIDs) < row.size:
        archIDs.extend(resourceManager.allocateArchLevel(granularity))
      for idx, x in enumerate(row):
        row[idx] = archIDs[idx].getGlobalID()
