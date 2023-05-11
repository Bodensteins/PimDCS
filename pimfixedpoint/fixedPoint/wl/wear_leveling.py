import numpy as np
import math
import warnings
from enum import Enum
import torch
import torch.nn.functional as F
import uuid
import pimfixedpoint.fixedPoint.nn.fixedPointArithmetic as fpA


class WearLevelingType(Enum):
  NotUse = "not-use"
  ColumnShift = "column-shift"
  RowShift = "row-shift"
  RowSwap = "row-swap"
  ColumnSwap = "column-swap"
  RowColumnShift = "row-column-shift"
  ZShift = "z-shift"

  def __str__(self):
    return self.value

class WearLevelingConfig():
  def __init__(self, intraWlType, useTIWL, intraPEShift, wlInterval, swapRatioOfTIWL, swapRatioOfArray, overlapUpdate,
               shiftNum=1):
    """
    Wear-leveling config.
    @param intraWlType: The intra-crossbar wear-leveling scheme.
    @param useTIWL: Uses TIWL or not.
    @param intraPEShift: Shifts crossbar within a PE or not.
    @param wlInterval: The interval of performing wear-leveling operation.
    @param swapRatioOfTIWL: PE Swaps ratio of TIWL.
    @param swapRatioOfArray: Physical array swap ratio of intra-crossbar wear-leveling schemes.
    @param overlapUpdate: Performs wear-leveling when writing new data to physical arrays. The additional writes
    introduced by wear-leveling would be ignored when using overlap update.
    @param shiftNum: The granularity of shifting schemes.
    """
    if not isinstance(intraWlType, WearLevelingType):
      raise TypeError("wlType must be an instance of WearLevelingType(Enum)")
    assert swapRatioOfTIWL > 0 and swapRatioOfTIWL <= 1
    assert swapRatioOfArray > 0 and swapRatioOfArray <= 1

    self.intraWlType = intraWlType
    self.useTIWL = useTIWL
    self.intraPEShift = intraPEShift
    self.wlInterval = wlInterval
    self.swapRatioOfTIWL = swapRatioOfTIWL  # Not support yet.
    self.swapRatioOfArray = swapRatioOfArray
    self.overlapUpdate = overlapUpdate
    self.shiftNum = shiftNum


class LogicalArray:
  def __init__(self, unique_key, logicalArraySize: tuple, physicalArraySize: tuple, splitBits: bool,
               bitWidth: int, cellBits: int, device: torch.device):
    """
    The logical array is a virtual reference of a matrix. If the size of logical array is larger than
    the size of physical array, the logical array would be divided into multiple sub-blocks and be
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
    self.intervalDiffCntList = [torch.zeros(self.logicalArraySize, dtype=torch.int64, device=self.device) for i in range(self.cellNumPerValue)]

    r = math.ceil(logicalArraySize[0] / physicalArraySize[0])
    if self.splitBits:
      c = math.ceil(logicalArraySize[1] / physicalArraySize[1]) * self.cellNumPerValue
    else:
      c = math.ceil(logicalArraySize[1] * bitWidth / (physicalArraySize[1] * cellBits))
    self.pidMap2D = np.full([r, c], np.iinfo(np.int64).max, dtype='int64')


  def updateDiffCntList(self, diff):
    for i in range(self.cellNumPerValue):
      for j in range(self.cellBits):
        self.intervalDiffCntList[i] += diff & 1
        diff = diff >> 1


  def resetDiffCntList(self, wlManager):
    # flush intervalDiffCntList
    if self.splitBits:
      colOffset = math.ceil(self.logicalArraySize[1] / self.physicalArraySize[1])
      for i in range(self.cellNumPerValue):
        row_indices = [] if self.physicalArraySize[0] > self.intervalDiffCntList[i].shape[0] else torch.arange(self.physicalArraySize[0], self.intervalDiffCntList[i].shape[0], self.physicalArraySize[0])
        for j, row_split in enumerate(torch.tensor_split(self.intervalDiffCntList[i], row_indices, dim=0)):
          col_indices = [] if self.physicalArraySize[1] > row_split.shape[1] else torch.arange(self.physicalArraySize[1], row_split.shape[1], self.physicalArraySize[1])
          for k, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
            col_idx = i * colOffset + k
            pid = self.pidMap2D[j][col_idx]
            wlManager.updateWriteCountDict(pid, col_split)
    else:
      countMap = torch.dstack(tuple(self.intervalDiffCntList)).view(
        [self.logicalArraySize[0], self.cellNumPerValue * self.logicalArraySize[1]])
      colChunkSize = math.ceil(countMap.shape[1] / self.pidMap2D.shape[1])
      row_indices = [] if self.physicalArraySize[0] > countMap.shape[0] else torch.arange(self.physicalArraySize[0], countMap.shape[0], self.physicalArraySize[0])
      for i, row_split in enumerate(torch.tensor_split(countMap, row_indices,dim=0)):
        col_indices = [] if colChunkSize > row_split.shape[1] else torch.arange(colChunkSize, row_split.shape[1], colChunkSize)
        for j, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
          pid = self.pidMap2D[i][j]
          wlManager.updateWriteCountDict(pid, col_split)

    # reset intervalDiffCntList
    self.intervalDiffCntList = [torch.zeros(self.logicalArraySize, dtype=torch.int64, device=self.device) for i in range(self.cellNumPerValue)]


class PhysicalArray:
  def __init__(self):
    pass


class PimWearLeveling:
  def __init__(self, wlConfig: WearLevelingConfig, physicalArraySize: tuple,
               cellBits: int = 1, splitBits: bool = False, arraysPerPE: int = 16, PEsPerTile: int = 32,
               tilesPerBank: int = 1, banksPerChip: int = 1, device=torch.device('cpu')):
    """
    Initiates a PIM wear-leveling manager.
    @param wlConfig: Wear-leveling config.
    @param physicalArraySize: The size of physical array.
    @param cellBits: Cell bits.
    @param splitBits: Splits data by bits and distributes them to different physical arrays.
    @param arraysPerPE: The number of physical arrays in a PE.
    @param PEsPerTile: The number of PEs in a tile.
    @param tilesPerBank: The number of tiles in a bank. (Not support yet!)
    @param banksPerChip: The number of banks in a chip. (Not support yet!)
    """
    self.stepCount = 0
    self.wlConfig = wlConfig

    # array config
    self.physicalArraySize = physicalArraySize
    self.cellBits = cellBits
    self.splitBits = splitBits  # split bits and store in different crossbars
    self.banksPerChip = banksPerChip
    self.TilesPerBank = tilesPerBank
    self.PEsPerTile = PEsPerTile
    self.arraysPerPE = arraysPerPE
    self.device = device

    self.logicalArrayDict = {}
    self.cellWriteCountDict = {}
    self.cellCurrentTensorDict = {}

    self.pid2lid = {}
    self.peid2pid = {}
    # physicalArrayInfo: ("pid", "pe_id", "iwc", "twc")
    self.physicalArrayInfo = np.empty((0, 4), dtype=np.uint64)
    # peInfo: ("pe_id", "is_full", "pe_iwc", "pe_twc")
    self.peInfo = np.empty((0, 4), dtype=np.uint64)

    self.current_pe_id = 0
    self.current_pid = 0

  def init_by_nn_module(self, dnn_module: torch.nn.Module):
    """
    Init wear-leveling manager by given pytorch nn module.
    @param dnn_module: pytorch nn module
    @return:
    """
    if not self.wlConfig.overlapUpdate:
      warnings.warn("You're training DNN module, please turn on overlapUpdate.")
      self.wlConfig.overlapUpdate = True
    for name, layer in dnn_module.named_modules():
      if hasattr(layer, 'fp_weight'):
        sz = layer.fp_weight.size()
        self.allocLogicalArray(name, sz, layer.weightBits)
    # if the final PE is not full
    if self.arraysPerPE > 0 and self.peInfo[self.current_pe_id - 1][1] == 0:
      # filled pe
      free_num = self.arraysPerPE - len(self.peid2pid[self.current_pe_id - 1])
      for i in range(free_num):
        unique_key = uuid.uuid4()
        self.logicalArrayDict[unique_key] = LogicalArray(unique_key, (1, 1), self.physicalArraySize, self.splitBits,
                                                         self.physicalArraySize[1], self.cellBits, self.device)
        pid, pe_id = self.allocPhysicalArray()
        self.logicalArrayDict[unique_key].pidMap2D[0, 0] = pid
        self.cellCurrentTensorDict[pid] = torch.zeros(self.physicalArraySize, dtype=torch.int64, device=self.device)
        self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
        self.pid2lid[pid] = unique_key


  def init_by_tensor_list(self, tensor_list, bitWidth):
    """
    Init wear-leveling manager by given tensors list.
    @param tensor_list: pytorch tensors list
    @param bitWidth:
    @return:
    """
    for idx, tensor in enumerate(tensor_list):
      self.allocLogicalArray(str(idx), tensor.shape, bitWidth=bitWidth)
    # if the final PE is not full
    if self.arraysPerPE > 1 and self.peInfo[-1][1] == 0:
      # filled pe
      free_num = self.arraysPerPE - len(self.peid2pid[self.current_pe_id])
      for i in range(free_num):
        unique_key = uuid.uuid4()
        self.logicalArrayDict[unique_key] = LogicalArray(unique_key, (1, 1), self.physicalArraySize, self.splitBits,
                                                         self.physicalArraySize[1], self.cellBits)
        pid, pe_id = self.allocPhysicalArray()
        self.logicalArrayDict[unique_key].pidMap2D[0, 0] = pid
        self.cellCurrentTensorDict[pid] = torch.zeros(self.physicalArraySize, dtype=torch.int64, device=self.device)
        self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
        self.pid2lid[pid] = unique_key



  def allocPhysicalArray(self):
    """
    Allocate a physical array.
    @return: PID and PE ID
    """
    target_pid = self.current_pid
    # find a free PE
    target_idx = np.where(self.peInfo[:, 1] == 0)[0]
    target_pe_id = target_idx[0] if target_idx.size > 0 else self.current_pe_id

    if not target_idx.size > 0:
      # allocate a new pe
      self.peInfo = np.append(self.peInfo, np.array([[target_pe_id, 0, 0, 0]], dtype=np.uint64), axis=0)
      self.peid2pid[target_pe_id] = []
      self.current_pe_id = self.current_pe_id + 1
    self.peid2pid[target_pe_id].append(target_pid)
    if len(self.peid2pid[target_pe_id]) == self.arraysPerPE:
      self.peInfo[target_pe_id][1] = 1

    self.current_pid = self.current_pid + 1
    self.cellWriteCountDict[target_pid] = torch.zeros(self.physicalArraySize, dtype=torch.int64, device=self.device)
    return target_pid, target_pe_id

  def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
    """
    Allocate a logical array.
    @param unique_key: unique key of the logical array
    @param logicalArraySize: size of logical array. [rowSize, colSize]
    @param bitWidth: bit width
    @return:
    """
    assert unique_key not in self.logicalArrayDict
    self.logicalArrayDict[unique_key] = LogicalArray(unique_key, logicalArraySize, self.physicalArraySize,
                                                     self.splitBits, bitWidth, self.cellBits, self.device)
    for row, col in np.ndindex(self.logicalArrayDict[unique_key].pidMap2D.shape):
      pid, pe_id = self.allocPhysicalArray()
      self.logicalArrayDict[unique_key].pidMap2D[row, col] = pid
      self.physicalArrayInfo = np.append(self.physicalArrayInfo, np.array([[pid, pe_id, 0, 0]], dtype=np.uint64), axis=0)
      self.pid2lid[pid] = unique_key


  def step(self, old_data_dict: dict, new_data_dict: dict):
    """
    Performs a single write step. The wear-leveling operation would be performed at wear-leveling
    intervals.
    @param old_data_dict: old matrix datas
    @param new_data_dict: new matrix datas
    @return:
    """
    # write logic array. record new data to get additional writes when performing wear-leveling
    for param_name in old_data_dict:
      self.writeLogicArray(param_name, old_data_dict[param_name], new_data_dict[param_name],
                           is_interval_end=True if self.stepCount % self.wlConfig.wlInterval == 0 else False)
    if self.stepCount % self.wlConfig.wlInterval == 0:
      # intra wear-leveling
      if self.wlConfig.intraWlType == WearLevelingType.ColumnShift or self.wlConfig.intraWlType == WearLevelingType.RowShift:
        self.shifting(not (self.wlConfig.useTIWL + self.wlConfig.intraPEShift))
      elif self.wlConfig.intraWlType == WearLevelingType.ColumnSwap or self.wlConfig.intraWlType == WearLevelingType.RowSwap:
        self.swapping(not (self.wlConfig.useTIWL + self.wlConfig.intraPEShift))
      elif self.wlConfig.intraWlType == WearLevelingType.RowColumnShift:
        self.rowColumnShift(not (self.wlConfig.useTIWL + self.wlConfig.intraPEShift))
      elif self.wlConfig.intraWlType == WearLevelingType.ZShift:
        if self.stepCount % self.physicalArraySize[0] == 0:
          self.rowColumnShift(not (self.wlConfig.useTIWL + self.wlConfig.intraPEShift))
        else:
          self.shifting(not (self.wlConfig.useTIWL + self.wlConfig.intraPEShift))
      # inter wear-leveling
      if self.wlConfig.useTIWL:
        self.TIWLByPE(not self.wlConfig.intraPEShift)
      if self.wlConfig.intraPEShift:
        self.intraArrayShift(True)

      # reset IWC
      self.physicalArrayInfo[:, 2] = 0
      self.peInfo[:, 2] = 0

    self.stepCount += 1



  def writeLogicArray(self, logical_key, old_data, new_data, is_interval_end=False):
    """
    Writes matrix to a logical array.
    @param logical_key: unique key of the logical array.
    @param old_data: old matrix
    @param new_data: new matrix
    @param is_interval_end: Record new data before performing wear-leveling operation.
    @return:
    """
    logicalArray = self.logicalArrayDict[logical_key]
    pidMap2D = logicalArray.pidMap2D
    cellNumPerValue = math.ceil(logicalArray.bitWidth / self.cellBits)
    diff = old_data ^ new_data
    logicalArray.updateDiffCntList(diff)

    # reset the interval diff count list and record new data
    if is_interval_end:
      # reset
      logicalArray.resetDiffCntList(self)

      # record new data
      new_bit_list = [torch.zeros_like(new_data, dtype=torch.int64, device=self.device) for i in range(cellNumPerValue)]
      mask = (1 << self.cellBits) - 1
      for i in range(cellNumPerValue):
        new_bit_list[i] = new_data & mask
        new_data = new_data >> self.cellBits
      if self.splitBits:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        colOffset = math.ceil(logicalArray.logicalArraySize[1] / self.physicalArraySize[1])
        for i in range(cellNumPerValue):
          row_indices = [] if self.physicalArraySize[0] > new_bit_list[i].shape[0] else torch.arange(self.physicalArraySize[0], new_bit_list[i].shape[0], self.physicalArraySize[0])
          for j, row_split in enumerate(torch.tensor_split(new_bit_list[i], row_indices, dim=0)):
            col_indices = [] if self.physicalArraySize[1] > row_split.shape[1] else torch.arange(self.physicalArraySize[1], row_split.shape[1], self.physicalArraySize[1])
            for k, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
              col_idx = i * colOffset + k
              pid = pidMap2D[j][col_idx]
              self.updateCellCurrentTensor(pid, col_split)
      else:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        record_data = torch.dstack(new_bit_list).view(new_data.shape[0], -1)
        colChunkSize = math.ceil(record_data.shape[1] / pidMap2D.shape[1])
        row_indices = [] if self.physicalArraySize[0] > record_data.shape[0] else torch.arange(self.physicalArraySize[0], record_data.shape[0], self.physicalArraySize[0])
        for i, row_split in enumerate(torch.tensor_split(record_data, row_indices, dim=0)):
          col_indices = [] if colChunkSize > row_split.shape[1] else torch.arange(colChunkSize, row_split.shape[1], colChunkSize)
          for j, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
            pid = pidMap2D[i][j]
            self.updateCellCurrentTensor(pid, col_split)

  def shifting(self, isFinalWL):
    """
    Performs shifting intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnShift else False
    for pid in self.physicalArrayInfo[:, 0]:
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = torch.roll(self.cellCurrentTensorDict[pid], shifts=(self.wlConfig.shiftNum),
                                dims=(1 if byColumn else 0))
        diff = shiftedTensor ^ self.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        # update current tensor dict for following wear-leveling operation
        self.updateCellCurrentTensor(pid, shiftedTensor)
      self.cellWriteCountDict[pid] = torch.roll(self.cellWriteCountDict[pid], shifts=(self.wlConfig.shiftNum),
                                             dims=(1 if byColumn else 0))

  def rowColumnShift(self, isFinalWL):
    """
    Performs row column shifting intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    for pid in self.physicalArrayInfo[:, 0]:
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = torch.roll(self.cellCurrentTensorDict[pid], shifts=(self.wlConfig.shiftNum, self.wlConfig.shiftNum),
                                dims=(0, 1))
        diff = shiftedTensor ^ self.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        # update current tensor dict for following wear-leveling operation
        self.updateCellCurrentTensor(pid, shiftedTensor)
      self.cellWriteCountDict[pid] = torch.roll(self.cellWriteCountDict[pid],
                                             shifts=(self.wlConfig.shiftNum, self.wlConfig.shiftNum),
                                             dims=(0, 1))

  def swapping(self, isFinalWL):
    """
    Performs swapping intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnSwap else False
    idxLen = self.physicalArraySize[1] if byColumn else self.physicalArraySize[0]
    for pid in self.physicalArrayInfo[:, 0]:
      swapNum = int((idxLen * self.wlConfig.swapRatioOfArray) / 2)
      oldIdx = torch.stack([torch.arange(idxLen), self.cellWriteCountDict[pid].sum(dim=0 if byColumn else 1)], dim=1)
      sortSum = oldIdx[oldIdx[:, 1].argsort()]
      for idx in range(swapNum):
        oldIdx[[sortSum[idx][0], sortSum[-idx - 1][0]]] = oldIdx[[sortSum[-idx - 1][0], sortSum[idx][0]]]

      if not self.wlConfig.overlapUpdate:
        swappedTensor = self.cellCurrentTensorDict[pid]
        if byColumn:
          swappedTensor = torch.take_along_dim(swappedTensor, torch.unsqueeze(oldIdx[:, 0], dim=0), dim=1)
        else:
          swappedTensor = torch.take_along_dim(swappedTensor, torch.unsqueeze(oldIdx[:, 0], dim=1), dim=0)
        diff = swappedTensor ^ self.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        self.updateCellCurrentTensor(pid, swappedTensor)

      if byColumn:
        self.cellWriteCountDict[pid] = torch.take_along_dim(self.cellWriteCountDict[pid],
                                                          torch.unsqueeze(oldIdx[:, 0], dim=0), dim=1)
      else:
        self.cellWriteCountDict[pid] = torch.take_along_dim(self.cellWriteCountDict[pid],
                                                          torch.unsqueeze(oldIdx[:, 0], dim=1), dim=0)


  def TIWLByPE(self, isFinalWL):
    """
    Performs table-based inter-crossbar wear-leveling scheme (TIWL).
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """

    def swapPhysicalArray(pid_a, pid_b):
      logical_key_a = self.pid2lid.pop(pid_a)
      logical_key_b = self.pid2lid.pop(pid_b)
      self.pid2lid[pid_a] = logical_key_b
      self.pid2lid[pid_b] = logical_key_a
      if logical_key_a in self.logicalArrayDict:
        idx_a = np.where(self.logicalArrayDict[logical_key_a].pidMap2D == pid_a)
        self.logicalArrayDict[logical_key_a].pidMap2D[idx_a] = pid_b
      if logical_key_b in self.logicalArrayDict:
        idx_b = np.where(self.logicalArrayDict[logical_key_b].pidMap2D == pid_b)
        self.logicalArrayDict[logical_key_b].pidMap2D[idx_b] = pid_a
      if not self.wlConfig.overlapUpdate:
        srcTensor = self.cellCurrentTensorDict[pid_a]
        diff = srcTensor ^ self.cellCurrentTensorDict[pid_b]
        self.updateWriteCountByDifference(pid_a, diff, isFinalWL)
        self.updateWriteCountByDifference(pid_b, diff, isFinalWL)
        self.updateCellCurrentTensor(pid_a, self.cellCurrentTensorDict[pid_b])
        self.updateCellCurrentTensor(pid_b, srcTensor)


    def swapPE(pe_id_a, pe_id_b):
      # step 1: find pid_list of pe
      pid_list_a = self.peid2pid[pe_id_a]
      pid_list_b = self.peid2pid[pe_id_b]
      # step 2: record new logical_key mapping for two pe
      for pid_a, pid_b in zip(pid_list_a, pid_list_b):
        swapPhysicalArray(pid_a, pid_b)

    # create new iwc and twc map
    sort_iwc = self.peInfo[self.peInfo[:, 2].argsort(), 0].tolist()
    sort_twc = np.flip(self.peInfo[self.peInfo[:, 3].argsort(), 0]).tolist()
    while len(sort_iwc) > 1:
      pe_id_a = sort_iwc[0]
      pe_id_b = sort_twc[0]
      if pe_id_a != pe_id_b:
        swapPE(pe_id_a, pe_id_b)
        sort_iwc.remove(pe_id_b)
        sort_twc.remove(pe_id_a)
      sort_iwc.remove(pe_id_a)
      sort_twc.remove(pe_id_b)

  def intraArrayShift(self, isFinalWL):
    """
    Shifts crossbar within a PE.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    # old_pid2lidDataFrame = self.pid2lidDataFrame.copy()
    for pe_id in self.peid2pid:
      new_pid_list = np.roll(self.peid2pid[pe_id], shift=(1), axis=(0))

      # backup the final one of old_pid_list
      last_logical_key_old = self.pid2lid[self.peid2pid[pe_id][-1]]
      pidMap2D_idx = np.where(self.logicalArrayDict[last_logical_key_old].pidMap2D == self.peid2pid[pe_id][-1])
      cellCurrentTensor = self.cellCurrentTensorDict[self.peid2pid[pe_id][-1]]

      for pid_idx in range(len(new_pid_list) - 1):
        # update self.pid2lidDataFrame
        old_logical_key = self.pid2lid[self.peid2pid[pe_id][pid_idx]]
        old_idx = np.where(self.logicalArrayDict[old_logical_key].pidMap2D == self.peid2pid[pe_id][pid_idx])

        # update self.logicalArrayDict
        self.pid2lid[new_pid_list[pid_idx]] = old_logical_key
        self.logicalArrayDict[old_logical_key].pidMap2D[old_idx] = new_pid_list[pid_idx]

        if not self.wlConfig.overlapUpdate:
          diff = self.cellCurrentTensorDict[new_pid_list[pid_idx]] ^ self.cellCurrentTensorDict[self.peid2pid[pe_id][pid_idx]]
          self.updateWriteCountByDifference(new_pid_list[pid_idx], diff, isFinalWL)
          self.updateCellCurrentTensor(new_pid_list[pid_idx], self.cellCurrentTensorDict[self.peid2pid[pe_id][pid_idx]])

      self.pid2lid[new_pid_list[-1]] = last_logical_key_old
      self.logicalArrayDict[last_logical_key_old].pidMap2D[pidMap2D_idx] = new_pid_list[-1]

      if not self.wlConfig.overlapUpdate:
        diff = self.cellCurrentTensorDict[new_pid_list[-1]] ^ cellCurrentTensor
        self.updateWriteCountByDifference(new_pid_list[-1], diff, isFinalWL)
        self.updateCellCurrentTensor(new_pid_list[-1], cellCurrentTensor)


  def updateWriteCountDict(self, pid, writeCnt):
    """
    Updates cell write count dict of wear-leveling manager.
    @param pid: PID of the physical array.
    @param writeCnt: cell writes count tensor of the physical array.
    @return:
    """
    if writeCnt.size(dim=0) < self.cellWriteCountDict[pid].size(dim=0) or writeCnt.size(dim=1) < \
        self.cellWriteCountDict[pid].size(dim=1):
      p2d = (0, self.cellWriteCountDict[pid].size(dim=1) - writeCnt.size(dim=1), 0, self.cellWriteCountDict[pid].size(dim=0) - writeCnt.size(dim=0))
      writeCnt = F.pad(writeCnt, pad=p2d, mode='constant', value=0)

    pe_id = self.physicalArrayInfo[pid][1]

    self.cellWriteCountDict[pid] += writeCnt
    writeCntSum = writeCnt.sum().item()

    self.physicalArrayInfo[pid][2:4] += writeCntSum
    self.peInfo[pe_id][2:4] += writeCntSum


  def updateCellCurrentTensor(self, pid, cellDataTensor):
    """
    Updates cell current tensor dict for wear-leveling manager.
    @param pid: PID of physical array.
    @param cellDataTensor: cell current tensor.
    @return:
    """
    if cellDataTensor.shape[0] < self.physicalArraySize[0] or cellDataTensor.shape[1] < self.physicalArraySize[1]:
      p2d = (0, self.physicalArraySize[1] - cellDataTensor.size(dim=1), 0, self.physicalArraySize[0] - cellDataTensor.size(dim=0))
      self.cellCurrentTensorDict[pid] = F.pad(cellDataTensor, pad=p2d, mode='constant', value=0)
    else:
      self.cellCurrentTensorDict[pid] = cellDataTensor

  def updateWriteCountByDifference(self, pid, diff, isFinalWL):
    """
    Updates writes count dict, IWC and TWC of specific physical array.
    @param pid: PID of physical array.
    @param diff: Difference tensor
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    if diff.sum() != 0:
      for i in range(self.cellBits):
        wrtCnt = diff & 1
        diff = diff >> 1
        self.cellWriteCountDict[pid] += wrtCnt
        wrtCntSum = wrtCnt.sum()
        if isFinalWL:
          pid_idx = np.where(self.physicalArrayInfo[:, 0] == pid)[0][0]
          pe_id = self.physicalArrayInfo[pid_idx][1]
          pe_id_idx = np.where(self.peInfo[:, 0] == pe_id)[0][0]
          self.physicalArrayInfo[pid_idx][3] += wrtCntSum
          self.peInfo[pe_id_idx][3] += wrtCntSum

# Utils
def copy_fixed_point_params(nn_model):
  params_dict = {}
  for name, layer in nn_model.named_modules():
    if hasattr(layer, 'weightBits') and hasattr(layer, 'fp_weight'):
      params_dict[name] = fpA.to_int(layer.fp_weight).clone()
  return params_dict