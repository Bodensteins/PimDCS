import math
import numpy as np
import torch
import torch.nn.functional as F
from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.resourceManager import ResourceManager
from src.pimtorch.simulator.mapping.mappingTable import MappingTable
from src.pimtorch.simulator.addressParser import ArchLevel, ArchID
import datetime
class LogicalTensor:
  def __init__(self, size: tuple, name: str, resourceManager: ResourceManager, mappingTable: MappingTable,
               allocator, granularity: ArchLevel, **kwargs):
    self.size = size
    self.name = name
    self.granularity = granularity
    self.dataSplitting = cfg.dataSplitting

    assert granularity == ArchLevel.crx or granularity == ArchLevel.ou
    if granularity == ArchLevel.crx:
      r = math.ceil(size[0] / cfg.crxShape[0])
      if self.dataSplitting:
        c = math.ceil(size[1] / cfg.crxShape[1]) * cfg.cellNumPerValue
      else:
        c = math.ceil(size[1] * cfg.weightBitWidth / (cfg.crxShape[1] * cfg.cellBits))
      self.cellMapShape = (r * cfg.crxShape[0], c * cfg.crxShape[1])
    else:
      r = math.ceil(size[0] / cfg.ouShape[0])
      if self.dataSplitting:
        c = math.ceil(size[1] / cfg.ouShape[1]) * cfg.cellNumPerValue
      else:
        c = math.ceil(size[1] * cfg.weightBitWidth / (cfg.ouShape[1] * cfg.cellBits))
      self.cellMapShape = (r * cfg.ouShape[0], c * cfg.ouShape[1])
    self.mappingShape = (r, c)
    self.mapping2D = np.full(self.mappingShape, -1, dtype=int)
    allocator.allocateLogicalTensor(mapping2D=self.mapping2D, resourceManager=resourceManager,
                                    granularity=getattr(ArchLevel, cfg.granularityAlloc), **kwargs)
    mappingTable.addLogicalTensor(self)

  def write(self, data):
    raise NotImplementedError("LogicalTensor is a basic class, the write() function is not implemented.")

  def getMapping2D(self):
    return self.mapping2D


  def __str__(self):
    return "LogicalTensor (%s)\n%s" % (self.name, self.mapping2D.__str__())


class LogicalTensorForWL(LogicalTensor):
  def __init__(self, size: tuple, name: str, resourceManager: ResourceManager, mappingTable: MappingTable,
               allocator, wearLevelingManager):
    super().__init__(size, name, resourceManager, mappingTable, allocator, ArchLevel.crx)
    self.wlManager = wearLevelingManager
    self.lastData = torch.zeros(size=size, dtype=cfg.torchInt, requires_grad=False)
    self.writeCountBuffer = torch.zeros(size=(cfg.cellNumPerValue, *size), dtype=torch.int32, requires_grad=False)
    self.lastCellBit = torch.zeros_like(self.writeCountBuffer, dtype=torch.int32)

    # padding for writing
    padding_r = cfg.crxShape[0] - (self.size[0] % cfg.crxShape[0])
    if self.dataSplitting:
      padding_c = cfg.crxShape[1] - (self.size[1] % cfg.crxShape[1])
    else:
      padding_c = self.cellMapShape[1] - (self.size[1] * cfg.weightBitWidth)
    self.paddingShape = (0, padding_c, 0, padding_r)

    # LASR
    crx_n_of_tile = cfg.crx_n * cfg.pe_n
    crx_n_of_bank = crx_n_of_tile * cfg.tile_n
    crx_n_of_stripe = math.ceil(self.mapping2D.shape[1] / cfg.crx_n) * cfg.crx_n
    row_n = math.floor(crx_n_of_tile / crx_n_of_stripe)
    if row_n >= 1:
      self.PEBlockShape = [math.ceil(self.mapping2D.shape[0] / row_n), 1]
    else:
      row_n_bank = math.floor(crx_n_of_bank / crx_n_of_stripe)
      assert row_n_bank >= 1
      self.PEBlockShape = [math.ceil(self.mapping2D.shape[0] / row_n_bank), 1]
    self.updatePEBlock()


  def write(self, data: torch.Tensor, flushWriteCounts = False, updateLastCellBit = True):
    diff = self.lastData ^ data
    self.lastData = data
    # assert flushWriteCounts + updateLastCellBit < 2, \
    #   "Update last cell bit should be performed before flushing write counts."
    if updateLastCellBit:
      self.lastCellBit.zero_()
      for i in range(cfg.cellNumPerValue):
        for j in range(cfg.cellBits):
          self.writeCountBuffer[i] += diff & 1
          self.lastCellBit[i] += data & 1
          diff.bitwise_right_shift_(1)
          data.bitwise_right_shift_(1)
      self.__updateLastCellBit__(self.lastCellBit)
    else:
      for i in range(cfg.cellNumPerValue):
        for j in range(cfg.cellBits):
          self.writeCountBuffer[i] += diff & 1
          diff.bitwise_right_shift_(1)

    if flushWriteCounts:
      self.__flushWriteCounts__()


  def updatePEBlock(self):
    self.PEBlock = {}
    for rowIdx, rowSplit in enumerate(np.array_split(self.mapping2D, self.PEBlockShape[0], axis=0)):
      for colIdx, colSplit in enumerate(np.array_split(rowSplit, self.PEBlockShape[1], axis=1)):
        for crxID in colSplit.flatten():
          peID = crxID // cfg.crx_n
          if peID not in self.PEBlock:
            self.PEBlock[peID] = self.name + "[%d][%d]" % (rowIdx, colIdx)

  def getCrxData(self, archID: ArchID):
    mappingIdx = np.where(self.mapping2D == archID.getArchLevelID(ArchLevel.crx))
    assert mappingIdx[0].size == 1, "Could not find specific crossbar (%s) in logical tensor (%s)" % (archID, self.name)
    return

  def __flushWriteCounts__(self):
    # flush intervalDiffCntList
    if self.dataSplitting:
      r = math.ceil(self.size[0] / cfg.crxShape[0])
      c = math.ceil(self.size[1] / cfg.crxShape[1])
      cellMap = F.pad(input=self.writeCountBuffer, pad=self.paddingShape, mode='constant', value=0)
      for i, bitSlice in enumerate(cellMap):
        for rowIdx in range(r):
          for colIdx in range(c):
            self.wlManager.updateWriteCounts(self.mapping2D[rowIdx][colIdx * cfg.cellNumPerValue + i],
                bitSlice[rowIdx * cfg.crxShape[0]: (rowIdx + 1) * cfg.crxShape[0],
                         colIdx * cfg.crxShape[1]: (colIdx + 1) * cfg.crxShape[1]])
    else:
      countMap = torch.vstack(tuple(self.writeCountBuffer.permute(2, 1, 0).transpose(1, 2))).transpose(0, 1)
      # padShape = (0, self.cellMapShape[1] - countMap.size(1), 0, self.cellMapShape[0] - countMap.size(0))
      cellMap = F.pad(input=countMap, pad=self.paddingShape, mode='constant', value=0)
      for idx, crxWriteCounts in enumerate(torch.tensor_split(cellMap, self.mappingShape[1], dim=1)):
        rowIdx = idx // self.mappingShape[1]
        colIdx = idx % self.mappingShape[1]
        self.wlManager.updateWriteCounts(self.mapping2D[rowIdx][colIdx], crxWriteCounts)

    # reset buffer
    self.writeCountBuffer.zero_()

  def __updateLastCellBit__(self, data: torch.Tensor):
    if self.dataSplitting:
      r = math.ceil(self.size[0] / cfg.crxShape[0])
      c = math.ceil(self.size[1] / cfg.crxShape[1])
      cellMap = F.pad(input=data, pad=self.paddingShape, mode='constant', value=0)
      for i, bitSlice in enumerate(cellMap):
        for rowIdx in range(r):
          for colIdx in range(c):
            self.wlManager.updateLastCellBit(self.mapping2D[rowIdx][colIdx * cfg.cellNumPerValue + i],
                                             bitSlice[rowIdx * cfg.crxShape[0]: (rowIdx + 1) * cfg.crxShape[0],
                                             colIdx * cfg.crxShape[1]: (colIdx + 1) * cfg.crxShape[1]])
    else:
      countMap = torch.vstack(tuple(data.permute(2, 1, 0).transpose(1, 2))).transpose(0, 1)
      # padShape = (0, self.cellMapShape[1] - countMap.size(1), 0, self.cellMapShape[0] - countMap.size(0))
      cellMap = F.pad(input=countMap, pad=self.paddingShape, mode='constant', value=0)
      for idx, crxWriteCounts in enumerate(torch.tensor_split(cellMap, self.mappingShape[1], dim=1)):
        rowIdx = idx // self.mappingShape[1]
        colIdx = idx % self.mappingShape[1]
        self.wlManager.updateLastCellBit(self.mapping2D[rowIdx][colIdx], crxWriteCounts)

