import numpy as np
import math
import warnings
from enum import Enum
import pandas as pd
import torch


class WearLevelingType(Enum):
  NotUse = 0
  ColumnShift = 1
  RowShift = 2
  RowSwapping = 3
  ColumnSwapping = 4
  RowColumnShift = 5


class WearLevelingConfig():
  def __init__(self, intraWlType, useTIWL, wlInterval, swapRatioOfTIWL, swapRatioOfArray, overlapUpdate, shiftNum=1):
    if not isinstance(intraWlType, WearLevelingType):
      raise TypeError("wlType must be an instance of WearLevelingType(Enum)")
    self.intraWlType = intraWlType
    self.useTIWL = useTIWL
    self.wlInterval = wlInterval
    assert swapRatioOfTIWL > 0 and swapRatioOfTIWL <= 1
    assert swapRatioOfArray > 0 and swapRatioOfArray <= 1
    self.swapRatioOfTIWL = swapRatioOfTIWL
    self.swapRatioOfArray = swapRatioOfArray
    self.overlapUpdate = overlapUpdate
    self.shiftNum = shiftNum


class LogicalArray:
  def __init__(self, unique_key, logicalArraySize: tuple, physicalArraySize: tuple, splitBits: bool,
               bitWidth: int, cellBits: int):
    self.unique_key = unique_key
    self.logicalArraySize = logicalArraySize
    self.physicalArraySize = physicalArraySize
    self.bitWidth = bitWidth
    self.splitBits = splitBits

    r = math.ceil(logicalArraySize[0] / physicalArraySize[0])
    if self.splitBits:
      cellNum = math.ceil(bitWidth / cellBits)
      c = math.ceil(logicalArraySize[1] / physicalArraySize[1]) * cellNum
    else:
      c = math.ceil(logicalArraySize[1] * bitWidth / (physicalArraySize[1] * cellBits))
    self.pidMap2D = np.full([r, c], np.nan, dtype='int64')


class PhysicalArray:
  def __init__(self):
    pass


class PimWearLeveling:
  def __init__(self, wlConfig: WearLevelingConfig, physicalArraySize: tuple,
               cellBits: int = 1, splitBits: bool = False, arraysPerPE: int = 16, PEsPerTile: int = 32,
               tilesPerBank: int = 1, banksPerChip: int = 1):
    self.scheduleCount = 0
    self.wlConfig = wlConfig

    # array config
    self.physicalArraySize = physicalArraySize
    self.cellBits = cellBits
    self.splitBits = splitBits  # split bits and store in different crossbars
    self.banksPerChip = banksPerChip
    self.TilesPerBank = tilesPerBank
    self.PEsPerTile = PEsPerTile
    self.arraysPerPE = arraysPerPE

    self.logicalArrayDict = {}
    self.cellWriteCountDict = {}
    self.cellCurrentTensorDict = {}

    self.pid2lidDataFrame = pd.DataFrame(columns=["pid", "logical_key", "pe_id", "iwc", "twc"])
    self.peDataFrame = pd.DataFrame(columns=["pe_id", "pid_list", "is_full", "pe_iwc", "pe_twc"])
    self.peDataFrame = self.peDataFrame.astype({"pid_list": object, "is_full": bool})

    self.current_pe_id = 0
    self.current_pid = 0

  def init_by_model(self, dnn_model: torch.nn.Module):
    if not self.wlConfig.overlapUpdate:
      warnings.warn("You're training DNN model, please turn on overlapUpdate.")
      self.wlConfig.overlapUpdate = True
    for name, layer in dnn_model.named_parameters():
      if hasattr(layer, 'weightBits') and hasattr(layer, 'fp_weight'):
        sz = layer.fp_weight.size()
        self.allocLogicalArray(name, sz, layer.weightBits)
    if not self.arraysPerPE > 0 and not self.peDataFrame.loc[self.peDataFrame['is_full'] == False].empty:
      # filled pe
      free_num = self.arraysPerPE - len(self.peDataFrame.loc[self.peDataFrame['is_full'] == False, 'pid_list'].tolist())
      for i in range(free_num):
        pid, pe_id = self.allocPhysicalArray()
        new_array = pd.DataFrame([[pid, "", pe_id, 0, 0]], columns=["pid", "logical_key", "pe_id", "iwc", "twc"])
        self.pid2lidDataFrame = pd.concat([self.pid2lidDataFrame, new_array])

  def init_by_tensor_list(self, tensor_list, bitWidth):
    for idx, tensor in enumerate(tensor_list):
      self.allocLogicalArray(str(idx), tensor.shape, bitWidth=bitWidth)
    if not self.arraysPerPE > 0 and not self.peDataFrame.loc[self.peDataFrame['is_full'] == False].empty:
      # filled pe
      free_num = self.arraysPerPE - len(self.peDataFrame.loc[self.peDataFrame['is_full'] == False, 'pid_list'].tolist())
      for i in range(free_num):
        pid, pe_id = self.allocPhysicalArray()
        new_array = pd.DataFrame([[pid, "", pe_id, 0, 0]], columns=["pid", "logical_key", "pe_id", "iwc", "twc"])
        self.pid2lidDataFrame = pd.concat([self.pid2lidDataFrame, new_array])

  def allocPhysicalArray(self):
    target_pe_id = self.current_pe_id
    target_pid = self.current_pid
    if not self.peDataFrame.loc[self.peDataFrame['is_full'] == False].empty:
      self.peDataFrame.iloc[target_pe_id]['pid_list'].append(target_pid)
      if len(self.peDataFrame.iloc[target_pe_id]['pid_list']) == self.arraysPerPE:
        self.peDataFrame.loc[self.peDataFrame['pe_id'] == target_pe_id, 'is_full'] = True
        self.current_pe_id = self.current_pe_id + 1
    else:
      new_pe = pd.DataFrame([[target_pe_id, [target_pid], False, 0, 0]],
                            columns=["pe_id", "pid_list", "is_full", "pe_iwc", "pe_twc"])
      self.peDataFrame = pd.concat([self.peDataFrame, new_pe])
    self.current_pid = self.current_pid + 1
    self.cellWriteCountDict[target_pid] = np.zeros(self.physicalArraySize, dtype=np.int64)
    return target_pid, target_pe_id

  def allocLogicalArray(self, unique_key, logicalArraySize, bitWidth):
    assert unique_key not in self.logicalArrayDict
    self.logicalArrayDict[unique_key] = LogicalArray(unique_key, logicalArraySize, self.physicalArraySize,
                                                     self.splitBits, bitWidth, self.cellBits)
    for row, col in np.ndindex(self.logicalArrayDict[unique_key].pidMap2D.shape):
      pid, pe_id = self.allocPhysicalArray()
      self.logicalArrayDict[unique_key].pidMap2D[row, col] = pid
      new_array = pd.DataFrame([[pid, unique_key, pe_id, 0, 0]], columns=["pid", "logical_key", "pe_id", "iwc", "twc"])
      self.pid2lidDataFrame = pd.concat([self.pid2lidDataFrame, new_array])

  def schedule(self, old_data_dict, new_data_dict):
    # write logic array. record new data to get additional writes when performing wear-leveling
    self.scheduleCount += 1
    for param_name in old_data_dict:
      self.writeLogicArray(param_name, old_data_dict[param_name], new_data_dict[param_name],
                           record_new_data=True if self.scheduleCount % self.wlConfig.wlInterval == 0 else False)

    if self.scheduleCount % self.wlConfig.wlInterval == 0:
      # intra wear-leveling
      if self.wlConfig.intraWlType == WearLevelingType.ColumnShift or self.wlConfig.intraWlType == WearLevelingType.RowShift:
        self.shifting()
      elif self.wlConfig.intraWlType == WearLevelingType.ColumnSwapping or self.wlConfig.intraWlType == WearLevelingType.RowSwapping:
        self.swapping()
      elif self.wlConfig.intraWlType == WearLevelingType.RowColumnShift:
        self.rowColumnShift()
      # inter wear-leveling
      if self.wlConfig.useTIWL:
        self.TIWLByPE()

      # reset IWC
      self.pid2lidDataFrame['iwc'].values[:] = 0
      self.peDataFrame['pe_iwc'].values[:] = 0

  def updateWriteCount(self, pid, writeCnt):
    if writeCnt.shape[0] < self.cellWriteCountDict[pid].shape[0] or writeCnt.shape[1] < \
        self.cellWriteCountDict[pid].shape[1]:
      pad = np.zeros_like(self.cellWriteCountDict[pid])
      pad[:writeCnt.shape[0], :writeCnt.shape[1]] = writeCnt
      writeCnt = pad
    pe_id = self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'pe_id'][0]
    self.cellWriteCountDict[pid] += writeCnt
    writeCntSum = writeCnt.sum()
    self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'iwc'] += writeCntSum
    self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'twc'] += writeCntSum
    self.peDataFrame.loc[self.peDataFrame['pe_id'] == pe_id, 'pe_iwc'] += writeCntSum
    self.peDataFrame.loc[self.peDataFrame['pe_id'] == pe_id, 'pe_twc'] += writeCntSum

  def updateCellCurrentTensor(self, pid, cellDataTensor):
    split_data_shape = [self.physicalArraySize[0], self.physicalArraySize[1] // self.cellBits]
    if cellDataTensor.shape[0] < split_data_shape[0] or cellDataTensor.shape[1] < split_data_shape[1]:
      pad = np.zeros(split_data_shape, dtype=np.int64)
      pad[:cellDataTensor.shape[0], :cellDataTensor.shape[1]] = cellDataTensor
      cellDataTensor = pad
    self.cellCurrentTensorDict[pid] = cellDataTensor


  def writeLogicArray(self, logical_key, old_data, new_data, record_new_data=False):
    logicalArray = self.logicalArrayDict[logical_key]
    pidMap2D = logicalArray.pidMap2D
    cellNumPerValue = math.ceil(logicalArray.bitWidth / self.cellBits)
    diff = old_data ^ new_data

    diffCntList = [np.zeros_like(diff, dtype=np.int64) for i in range(cellNumPerValue)]
    for i in range(cellNumPerValue):
      for j in range(self.cellBits):
        diffCntList[i] += diff & 1
        diff = diff >> 1

    if self.splitBits:
      colOffset = math.ceil(logicalArray.logicalArraySize[1] / self.physicalArraySize[1])
      for i in range(cellNumPerValue):
        for j, row_split in enumerate(np.array_split(diffCntList[i], pidMap2D.shape[0], axis=0)):
          for k, col_split in enumerate(np.array_split(row_split, pidMap2D.shape[1], axis=1)):
            col_idx = i * colOffset + k
            pid = pidMap2D[i][col_idx]
            self.updateWriteCount(pid, col_split)
    else:
      countMap = np.dstack(tuple(diffCntList)).reshape([logicalArray.logicalArraySize[0], cellNumPerValue * logicalArray.logicalArraySize[1]])
      for i, row_split in enumerate(np.array_split(countMap, pidMap2D.shape[0], axis=0)):
        for j, col_split in enumerate(np.array_split(row_split, pidMap2D.shape[1], axis=1)):
          pid = pidMap2D[i][j]
          self.updateWriteCount(pid, col_split)

    if record_new_data:
      new_bit_list = [np.zeros_like(new_data, dtype=np.int64) for i in range(cellNumPerValue)]
      for i in range(cellNumPerValue):
        for j in range(self.cellBits):
          new_bit_list[i] += new_data & 1
          new_data = new_data >> 1
      if self.splitBits:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        colOffset = math.ceil(logicalArray.logicalArraySize[1] / self.physicalArraySize[1])
        for i in range(cellNumPerValue):
          for j, row_split in enumerate(np.array_split(new_bit_list[i], pidMap2D.shape[0], axis=0)):
            for k, col_split in enumerate(np.array_split(row_split, pidMap2D.shape[1], axis=1)):
              col_idx = i * colOffset + k
              pid = pidMap2D[i][col_idx]
              self.updateCellCurrentTensor(pid, col_split)
      else:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        record_data = np.stack(new_bit_list).reshape(new_data.shape[0], -1)
        for i, row_split in enumerate(np.array_split(record_data, pidMap2D.shape[0], axis=0)):
          for j, col_split in enumerate(np.array_split(row_split, pidMap2D.shape[1], axis=1)):
            pid = pidMap2D[i][j]
            self.updateCellCurrentTensor(pid, col_split)


  def shifting(self):
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnShift else False
    for pid in self.pid2lidDataFrame['pid'].tolist():
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = np.roll(self.cellCurrentTensorDict[pid], shift=(self.wlConfig.shiftNum),
                                axis=(1 if byColumn else 0))
        diff = shiftedTensor ^ self.cellCurrentTensorDict[pid]
        for i in range(self.cellBits):
          wrtCnt = diff & 1
          diff = diff >> 1
          self.cellWriteCountDict[pid] += wrtCnt
          self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'twc'] += wrtCnt.sum()
      self.cellWriteCountDict[pid] = np.roll(self.cellWriteCountDict[pid], shift=(self.wlConfig.shiftNum),
                                             axis=(1 if byColumn else 0))

  def rowColumnShift(self):
    for pid in self.pid2lidDataFrame['pid'].tolist():
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = np.roll(self.cellCurrentTensorDict[pid], shift=(self.wlConfig.shiftNum, self.wlConfig.shiftNum),
                                axis=(0, 1))
        diff = shiftedTensor ^ self.cellCurrentTensorDict[pid]
        for i in range(self.cellBits):
          wrtCnt = diff & 1
          diff = diff >> 1
          self.cellWriteCountDict[pid] += wrtCnt
          self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'twc'] += wrtCnt.sum()
      self.cellWriteCountDict[pid] = np.roll(self.cellWriteCountDict[pid], shift=(self.wlConfig.shiftNum, self.wlConfig.shiftNum),
                                             axis=(0, 1))

  def swapping(self):
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnSwapping else False
    for pid in self.pid2lidDataFrame['pid'].tolist():
      if not self.wlConfig.overlapUpdate:
        swappedTensor = self.cellCurrentTensorDict[pid].copy()
      if byColumn:
        swapNum = (self.physicalArraySize[1] * self.wlConfig.swapRatioOfArray) // 2
        oldIdx = np.stack([np.arange(self.physicalArraySize[1]), self.cellWriteCountDict[pid].sum(dim=0)], axis=1,
                          dtype=np.int64)
        sortSum = oldIdx[oldIdx[:, 1].argsort()]

        for col in swapNum:
          oldIdx[:, [sortSum[col][0], sortSum[-col - 1][0]]] = oldIdx[:, [sortSum[-col - 1][0], sortSum[col][0]]]
          if not self.wlConfig.overlapUpdate:
            swappedTensor[:, [sortSum[col][0], sortSum[-col - 1][0]]] = swappedTensor[:,
                                                                        [sortSum[-col - 1][0], sortSum[col][0]]]
      else:
        swapNum = (self.physicalArraySize[0] * self.wlConfig.swapRatioOfArray) // 2
        oldIdx = np.stack([np.arange(self.physicalArraySize[0]), self.cellWriteCountDict[pid].sum(dim=1)], axis=1,
                          dtype=np.int64)
        sortSum = oldIdx[oldIdx[:, 1].argsort()]
        for row in swapNum:
          oldIdx[[sortSum[row][0], sortSum[-row - 1][0]]] = oldIdx[[sortSum[-row - 1][0], sortSum[row][0]]]
          if not self.wlConfig.overlapUpdate:
            swappedTensor[[sortSum[row][0], sortSum[-row - 1][0]]] = swappedTensor[
              [sortSum[-row - 1][0], sortSum[row][0]]]

      if not self.wlConfig.overlapUpdate:
        diff = swappedTensor ^ self.cellCurrentTensorDict[pid]
        for i in range(self.cellBits):
          wrtCnt = diff & 1
          diff = diff >> 1
          self.cellWriteCountDict[pid] += wrtCnt
          self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid, 'twc'] += wrtCnt.sum()
        self.cellWriteCountDict[pid] = torch.index_select(self.cellWriteCountDict[pid], dim=1 if byColumn else 0,
                                                          index=torch.LongTensor(oldIdx))

  def TIWLByPE(self):
    def swapPhysicalArray(pid_a, pid_b):
      def additialWrite(src_pid, dst_pid):
        srcTensor = self.cellCurrentTensorDict[src_pid].copy()
        diff = srcTensor ^ self.cellCurrentTensorDict[dst_pid]
        for i in range(self.cellBits):
          wrtCnt = diff & 1
          diff = diff >> 1
          self.cellWriteCountDict[src_pid] += wrtCnt
          self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == src_pid, 'twc'] += wrtCnt.sum()

      logical_key_a = self.pid2lidDataFrame[self.pid2lidDataFrame['pid'] == pid_a]['logical_key'][0]
      logical_key_b = self.pid2lidDataFrame[self.pid2lidDataFrame['pid'] == pid_b]['logical_key'][0]
      self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid_a, 'logical_key'] = logical_key_b
      self.pid2lidDataFrame.loc[self.pid2lidDataFrame['pid'] == pid_b, 'logical_key'] = logical_key_a
      idx_a = np.where(self.logicalArrayDict[logical_key_a].pidMap2D == pid_a)
      idx_b = np.where(self.logicalArrayDict[logical_key_a].pidMap2D == pid_b)
      self.logicalArrayDict[logical_key_a].pidMap2D[idx_a] = pid_b
      self.logicalArrayDict[logical_key_a].pidMap2D[idx_b] = pid_a
      if not self.wlConfig.overlapUpdate:
        additialWrite(pid_a, pid_b)
        additialWrite(pid_b, pid_a)

    def swapPE(pe_id_a, pe_id_b):
      # step 1: find pid_list of pe
      pid_list_a = self.peDataFrame[self.peDataFrame['pe_id'] == pe_id_a]['pid_list'][0]
      pid_list_b = self.peDataFrame[self.peDataFrame['pe_id'] == pe_id_b]['pid_list'][0]
      # step 2: record new logical_key mapping for two pe
      for pid_a, pid_b in zip(pid_list_a, pid_list_b):
        swapPhysicalArray(pid_a, pid_b)

    # create new iwc and twc map
    sort_iwc = self.peDataFrame[['pe_id', 'pe_iwc']].sort_values(by=['pe_iwc'])['pe_id'].to_list()
    sort_twc = self.peDataFrame[['pe_id', 'pe_twc']].sort_values(by=['pe_twc'], ascending=False)['pe_id'].to_list()
    while len(sort_iwc) > 1:
      pe_id_a = sort_iwc[0]
      pe_id_b = sort_twc[0]
      if pe_id_a != pe_id_b:
        swapPE(pe_id_a, pe_id_b)
        sort_iwc.remove(pe_id_b)
        sort_twc.remove(pe_id_a)
      sort_iwc.remove(pe_id_a)
      sort_twc.remove(pe_id_b)
