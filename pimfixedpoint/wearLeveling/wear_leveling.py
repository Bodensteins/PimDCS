import numpy as np
import json
from json import JSONEncoder
import math
import warnings
from enum import Enum
from collections import Counter
import torch
import torch.nn.functional as F
import uuid
import pimfixedpoint.fixedPoint.nn.fixedPointArithmetic as fpA
from pimfixedpoint.mapping.archConst import archConst
from pimfixedpoint.mapping.archMapping import SaveSpaceMapStrategy, ArchLevel, Arch
from pimfixedpoint.wearLeveling.nn.communication import SequentialCommModule

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
  def __init__(self, intraWlType, useTIWL, intraPEShift, wlInterval, PEShiftInterval, xbarShiftInterval,
               topK, swapRatioOfTIWL, swapRatioOfArray, overlapUpdate, saveInterval, savePath, shiftNum=1,
               store_transpose=True, store_json_mapping=False):
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
    self.PEShiftInterval = PEShiftInterval
    self.xbarShiftInterval = xbarShiftInterval
    self.swapRatioOfTIWL = swapRatioOfTIWL  # Not support yet.
    self.swapRatioOfArray = swapRatioOfArray
    self.overlapUpdate = overlapUpdate
    self.shiftNum = shiftNum
    self.saveInterval = saveInterval
    self.savePath = savePath
    self.topK = topK
    self.topKSize = 0
    self.store_transpose = store_transpose
    self.store_json_mapping = store_json_mapping

class PimWearLeveling:
  def __init__(self, mapping: SaveSpaceMapStrategy, wlConfig: WearLevelingConfig):
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
    self.mapping = mapping
    self.topKCount = 0
    self.mappingState = {}
    self.arch = self.mapping.arch
    self.comm_model = None
    self.comm_data = []

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
        self.mapping.allocLogicalArray(name, sz, layer.weightBits)
        # store transpose of weight
        if self.wlConfig.store_transpose:
          self.mapping.allocLogicalArray(name+".t", (sz[1], sz[0]), layer.weightBits)
    # self.wlConfig.topKSize = int(self.mapping.peInfo.shape[0] / 2 * self.wlConfig.topK * 0.01)
    self.wlConfig.topKSize = int(self.mapping.peInfo.shape[0] * self.wlConfig.topK * 0.01)
    self.comm_model = SequentialCommModule(arch=self.arch, model=dnn_module, archMapping=self.mapping,
                                           store_transpose=self.wlConfig.store_transpose)
    self.comm_analysis()


  def init_by_tensor_list(self, tensor_list, bitWidth):
    """
    Init wear-leveling manager by given tensors list.
    @param tensor_list: pytorch tensors list
    @param bitWidth:
    @return:
    """
    unique_keys = []
    for idx, tensor in enumerate(tensor_list):
      unique_key = str(uuid.uuid4())
      self.mapping.allocLogicalArray(unique_key, tensor.shape, bitWidth=bitWidth)
      unique_keys.append(unique_key)
    # self.wlConfig.topKSize = int(self.mapping.peInfo.shape[0] / 2 * self.wlConfig.topK * 0.01)
    self.wlConfig.topKSize = int(self.mapping.peInfo.shape[0] * self.wlConfig.topK * 0.01)
    return unique_keys


  def step(self, old_data_dict: dict, new_data_dict: dict):
    """
    Performs a single write step. The wear-leveling operation would be performed at wear-leveling
    intervals.
    @param old_data_dict: old matrix datas
    @param new_data_dict: new matrix datas
    @return:
    """
    # write logic array. record new data to get additional writes when performing wear-leveling
    wl_num = 0
    if self.stepCount % self.wlConfig.wlInterval == 0:
      wl_num += 1
    if self.stepCount % self.wlConfig.PEShiftInterval == 0:
      wl_num += 1
    if self.stepCount % self.wlConfig.xbarShiftInterval == 0:
      wl_num += 1

    if self.stepCount % self.wlConfig.wlInterval == 0 or self.stepCount % self.wlConfig.xbarShiftInterval == 0:
      is_interval_end = True
    elif self.wlConfig.intraPEShift and self.stepCount % self.wlConfig.PEShiftInterval == 0:
      is_interval_end = True
    else:
      is_interval_end = False

    for param_name in old_data_dict:
      self.writeLogicArray(param_name, old_data_dict[param_name], new_data_dict[param_name], is_interval_end=is_interval_end)
      if self.wlConfig.store_transpose:
        self.writeLogicArray(param_name+".t", old_data_dict[param_name].t(), new_data_dict[param_name].t(), is_interval_end=is_interval_end)

    if wl_num > 0 and self.wlConfig.intraPEShift and (self.stepCount % self.wlConfig.PEShiftInterval == 0):
      wl_num -= 1
      self.intraArrayShift(isFinalWL=(wl_num == 0))

    if wl_num > 0 and self.stepCount % self.wlConfig.xbarShiftInterval == 0:
      wl_num -= 1
      # intra wear-leveling
      if self.wlConfig.intraWlType == WearLevelingType.ColumnShift or self.wlConfig.intraWlType == WearLevelingType.RowShift:
        self.shifting(isFinalWL=(wl_num == 0))
      elif self.wlConfig.intraWlType == WearLevelingType.ColumnSwap or self.wlConfig.intraWlType == WearLevelingType.RowSwap:
        self.swapping(isFinalWL=(wl_num == 0))
      elif self.wlConfig.intraWlType == WearLevelingType.RowColumnShift:
        self.rowColumnShift(isFinalWL=(wl_num == 0))
      elif self.wlConfig.intraWlType == WearLevelingType.ZShift:
        if self.stepCount % archConst.phyArraySize[0] == 0:
          self.rowColumnShift(isFinalWL=(wl_num == 0))
        else:
          self.shifting(isFinalWL=(wl_num == 0))

    if wl_num > 0 and self.stepCount % self.wlConfig.wlInterval == 0:
      wl_num -= 1
      # inter wear-leveling
      if self.wlConfig.useTIWL:
        self.TIWLByPE(isFinalWL=(wl_num == 0))

      # reset IWC
      self.mapping.physicalArrayInfo[:, 2] = 0
      self.mapping.peInfo[:, 1] = 0

      # record mapping changes
      self.mappingState[self.stepCount] = {}
      for name, logicalArray in self.mapping.logicalArrayDict.items():
        self.mappingState[self.stepCount][name] = logicalArray.pidMap2D

      self.comm_analysis()
    if self.stepCount % self.wlConfig.saveInterval == 0:
      self.save_wl_data(filepath=self.wlConfig.savePath + "_wl_%d.pkl" % self.stepCount)
    self.stepCount += 1


  def comm_analysis(self):
    if self.comm_model:
      self.comm_model.reset_mapping(mappingState=None)
      addr_tensor = fpA.to_float(torch.tensor([0, ArchLevel.off_chip_level], dtype=torch.int64))
      f_out_addr_tensor = self.comm_model(addr_tensor)
      if self.wlConfig.store_transpose:
        f_out_addr_tensor.backward(f_out_addr_tensor)
      self.comm_data.append(self.comm_model.get_comm_tensor())

  def writeLogicArray(self, logical_key, old_data, new_data, is_interval_end=False):
    """
    Writes matrix to a logical array.
    @param logical_key: unique key of the logical array.
    @param old_data: old matrix
    @param new_data: new matrix
    @param is_interval_end: Record new data before performing wear-leveling operation.
    @return:
    """
    logicalArray = self.mapping.logicalArrayDict[logical_key]
    pidMap2D = logicalArray.pidMap2D
    cellNumPerValue = math.ceil(logicalArray.bitWidth / archConst.cellBits)
    # diff = torch.transpose(old_data ^ new_data, 0, 1)
    diff = torch.transpose(old_data ^ new_data, 0, 1).to(device=torch.device("cpu"))
    logicalArray.updateDiffCntList(diff)

    # reset the interval diff count list and record new data
    if is_interval_end:
      # reset
      logicalArray.resetDiffCntList(self)

      # record new data
      new_bit_list = [None for i in range(cellNumPerValue)]
      mask = (1 << archConst.cellBits) - 1
      for i in range(cellNumPerValue):
        # new_bit_list[i] = torch.transpose(new_data & mask, 0, 1)
        new_bit_list[i] = torch.transpose(new_data & mask, 0, 1).to(device=torch.device("cpu"))
        new_data = new_data >> archConst.cellBits
      if self.mapping.splitBits:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        # colOffset = math.ceil(logicalArray.logicalArraySize[1] / archConst.phyArraySize[1])
        for i in range(cellNumPerValue):
          row_indices = [] if archConst.phyArraySize[0] > new_bit_list[i].shape[0] else torch.arange(archConst.phyArraySize[0], new_bit_list[i].shape[0], archConst.phyArraySize[0])
          for j, row_split in enumerate(torch.tensor_split(new_bit_list[i], row_indices, dim=0)):
            col_indices = [] if archConst.phyArraySize[1] > row_split.shape[1] else torch.arange(archConst.phyArraySize[1], row_split.shape[1], archConst.phyArraySize[1])
            for k, col_split in enumerate(torch.tensor_split(row_split, col_indices, dim=1)):
              # col_idx = i * colOffset + k
              col_idx = k * cellNumPerValue + i
              pid = pidMap2D[j][col_idx]
              self.updateCellCurrentTensor(pid, col_split)
      else:
        # record_data shape: [logicRowSize, logicColSize * cellNumPerValue]
        record_data = torch.dstack(new_bit_list).view(logicalArray.logicalArraySize[0], -1)
        colChunkSize = math.ceil(record_data.shape[1] / pidMap2D.shape[1])
        row_indices = [] if archConst.phyArraySize[0] > record_data.shape[0] else torch.arange(archConst.phyArraySize[0], record_data.shape[0], archConst.phyArraySize[0])
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
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnShift \
                       or self.wlConfig.intraWlType == WearLevelingType.ZShift else False
    for pid in self.mapping.physicalArrayInfo[:, 0]:
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = torch.roll(self.mapping.cellCurrentTensorDict[pid], shifts=(self.wlConfig.shiftNum),
                                dims=(1 if byColumn else 0))
        diff = shiftedTensor ^ self.mapping.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        # update current tensor dict for following wear-leveling operation
        self.updateCellCurrentTensor(pid, shiftedTensor)
      self.mapping.cellWriteCountDict[pid] = torch.roll(self.mapping.cellWriteCountDict[pid],
                                                        shifts=(self.wlConfig.shiftNum), dims=(1 if byColumn else 0))

  def rowColumnShift(self, isFinalWL):
    """
    Performs row column shifting intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    for pid in self.mapping.physicalArrayInfo[:, 0]:
      if not self.wlConfig.overlapUpdate:
        shiftedTensor = torch.roll(self.mapping.cellCurrentTensorDict[pid],
                                   shifts=(self.wlConfig.shiftNum, self.wlConfig.shiftNum), dims=(0, 1))
        diff = shiftedTensor ^ self.mapping.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        # update current tensor dict for following wear-leveling operation
        self.updateCellCurrentTensor(pid, shiftedTensor)
      self.mapping.cellWriteCountDict[pid] = torch.roll(self.mapping.cellWriteCountDict[pid],
                                             shifts=(self.wlConfig.shiftNum, self.wlConfig.shiftNum), dims=(0, 1))

  def swapping(self, isFinalWL):
    """
    Performs swapping intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    byColumn = True if self.wlConfig.intraWlType == WearLevelingType.ColumnSwap else False
    idxLen = archConst.phyArraySize[1] if byColumn else archConst.phyArraySize[0]
    for pid in self.mapping.physicalArrayInfo[:, 0]:
      swapNum = int((idxLen * self.wlConfig.swapRatioOfArray) / 2)
      oldIdx = torch.stack([torch.arange(idxLen), self.mapping.cellWriteCountDict[pid].sum(dim=0 if byColumn else 1)], dim=1)
      sortSum = oldIdx[oldIdx[:, 1].argsort()]
      for idx in range(swapNum):
        oldIdx[[sortSum[idx][0], sortSum[-idx - 1][0]]] = oldIdx[[sortSum[-idx - 1][0], sortSum[idx][0]]]

      if not self.wlConfig.overlapUpdate:
        swappedTensor = self.mapping.cellCurrentTensorDict[pid]
        if byColumn:
          swappedTensor = torch.take_along_dim(swappedTensor, torch.unsqueeze(oldIdx[:, 0], dim=0), dim=1)
        else:
          swappedTensor = torch.take_along_dim(swappedTensor, torch.unsqueeze(oldIdx[:, 0], dim=1), dim=0)
        diff = swappedTensor ^ self.mapping.cellCurrentTensorDict[pid]
        self.updateWriteCountByDifference(pid, diff, isFinalWL)
        self.updateCellCurrentTensor(pid, swappedTensor)

      if byColumn:
        self.mapping.cellWriteCountDict[pid] = torch.take_along_dim(self.mapping.cellWriteCountDict[pid],
                                                          torch.unsqueeze(oldIdx[:, 0], dim=0), dim=1)
      else:
        self.mapping.cellWriteCountDict[pid] = torch.take_along_dim(self.mapping.cellWriteCountDict[pid],
                                                          torch.unsqueeze(oldIdx[:, 0], dim=1), dim=0)


  def TIWLByPE(self, isFinalWL):
    """
    Performs table-based inter-crossbar wear-leveling scheme (TIWL).
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    def swapPhysicalArray(pid_a, pid_b):
      logical_key_a = self.mapping.pid2lid.pop(pid_a)
      logical_key_b = self.mapping.pid2lid.pop(pid_b)
      self.mapping.pid2lid[pid_a] = logical_key_b
      self.mapping.pid2lid[pid_b] = logical_key_a
      idx_a = np.where(self.mapping.logicalArrayDict[logical_key_a].pidMap2D == pid_a)
      idx_b = np.where(self.mapping.logicalArrayDict[logical_key_b].pidMap2D == pid_b)
      self.mapping.logicalArrayDict[logical_key_a].pidMap2D[idx_a] = pid_b
      self.mapping.logicalArrayDict[logical_key_b].pidMap2D[idx_b] = pid_a

      if not self.wlConfig.overlapUpdate:
        srcTensor = self.mapping.cellCurrentTensorDict[pid_a]
        diff = srcTensor ^ self.mapping.cellCurrentTensorDict[pid_b]
        self.updateWriteCountByDifference(pid_a, diff, isFinalWL)
        self.updateWriteCountByDifference(pid_b, diff, isFinalWL)
        self.updateCellCurrentTensor(pid_a, self.mapping.cellCurrentTensorDict[pid_b])
        self.updateCellCurrentTensor(pid_b, srcTensor)

    def swapPE(pe_id_a, pe_id_b):
      # step 1: find pid_list of pe
      pid_list_a = self.mapping.peid2pid[pe_id_a]
      pid_list_b = self.mapping.peid2pid[pe_id_b]
      # step 2: record new logical_key mapping for two pe
      for pid_a, pid_b in zip(pid_list_a, pid_list_b):
        swapPhysicalArray(pid_a, pid_b)

    def getPEStripe():
      PEStripe = {}
      pe_shift_bits = self.arch._mask_size_list[ArchLevel.pe_level - 1]
      for unique_key, logicalArray in self.mapping.logicalArrayDict.items():
        for si, stripe in enumerate(logicalArray.pidMap2D):
          global_pe_id_list = list(map(lambda pid: (pid >> pe_shift_bits) << pe_shift_bits, stripe))
          stripe_width = self.arch.crx_n
          for idx, pid in enumerate(stripe):
            pe_id = global_pe_id_list[idx] >> pe_shift_bits
            if pe_id not in PEStripe:
              if self.arch.crx_n > 8:
                # Column Stripe
                stripe_col_idx = idx // stripe_width
                PEStripe[pe_id] = unique_key + ".%d" % stripe_col_idx
              else:
                # Row Stripe
                PEStripe[pe_id] = unique_key + ".%d" % si
      return PEStripe

    def getPERowColStripe():
      rowStripe = {}
      colStripe = {}
      pe_shift_bits = self.arch._mask_size_list[ArchLevel.pe_level - 1]
      for unique_key, logicalArray in self.mapping.logicalArrayDict.items():
        for si, stripe in enumerate(logicalArray.pidMap2D):
          stripe_width = self.arch.crx_n
          for idx, pid in enumerate(stripe):
            pe_id = pid >> pe_shift_bits
            if pe_id not in colStripe:
              # Column Stripe
              stripe_col_idx = idx // stripe_width
              colStripe[pe_id] = unique_key + ".%d" % stripe_col_idx
            if pe_id not in rowStripe:
              # Row Stripe
              rowStripe[pe_id] = unique_key + ".%d" % si
      return rowStripe, colStripe

    def getPEBlock():
      PEBlock = {}
      pe_shift_bits = self.arch._mask_size_list[ArchLevel.pe_level - 1]
      crx_n_of_tile = self.arch.crx_n * self.arch.pe_n
      crx_n_of_bank = crx_n_of_tile * self.arch.tile_n
      for unique_key, logicalArray in self.mapping.logicalArrayDict.items():
        mat_shape = logicalArray.pidMap2D.shape
        if not unique_key.startswith("Placeholder"):
          # 1. 计算pe block大小
          # Row Stripe
          # 如果Tile能放下整个行条带，尝试尽可能多放行条带，block大小: [floor(tile内xbar数量 / 当前行条带xbar数量)， pidMap列数] (<=)
          # 如果Tile不能放下整个行条带，放一个Bank，block大小: [floor(bank内xbar数量 / 当前行条带xbar数量)， pidMap列数] (>)
          crx_n_of_stripe = math.ceil(mat_shape[1] / self.arch.crx_n) * self.arch.crx_n
          row_n = math.floor(crx_n_of_tile / crx_n_of_stripe)
          if row_n >= 1:
            block_shape = [math.ceil(mat_shape[0] / row_n), 1]
          else:
            row_n_bank = math.floor(crx_n_of_bank / crx_n_of_stripe)
            assert row_n_bank >= 1
            block_shape = [math.ceil(mat_shape[0] / row_n_bank), 1]
        else:
          block_shape = [1, 1]
        # 2. 遍历pid
        for row_idx, row_split in enumerate(np.array_split(logicalArray.pidMap2D, block_shape[0], axis=0)):
          for col_idx, col_split in enumerate(np.array_split(row_split, block_shape[1], axis=1)):
            for pid in col_split.flatten():
              pe_id = pid >> pe_shift_bits
              if pe_id not in PEBlock:
                PEBlock[pe_id] = unique_key + "[%d][%d]" % (row_idx, col_idx)
      return PEBlock

    def swapSuitablePEs(iwc_stripe_id, twc_tile_bank_id):
      iwc_unzip = list(zip(*iwc_stripe_id))
      twc_unzip = list(zip(*twc_tile_bank_id))
      iwc_common_counter = Counter(iwc_unzip[1]).most_common()
      twc_common_tile = Counter(twc_unzip[1]).most_common()
      twc_common_bank = Counter(twc_unzip[2]).most_common()
      iwc_pe_id = []
      twc_pe_id = []
      # Placeholder should be processed at the end.
      if len(iwc_common_counter) > 1 and iwc_common_counter[0][0].startswith("Placeholder"):
        iwc_common_stripe = iwc_common_counter[1]
      else:
        iwc_common_stripe = iwc_common_counter[0]
      for pe_tuple in iwc_stripe_id:
        if len(iwc_pe_id) < iwc_common_stripe[1]:
          if pe_tuple[1] == iwc_common_stripe[0]:
            iwc_pe_id.append(pe_tuple[0])
        else:
          break
      for bank_id, bank_pe_cnt in twc_common_bank:
        if len(twc_pe_id) == len(iwc_pe_id):
          break
        bank_range = self.arch.get_same_level_pid_list(bank_id, ArchLevel.bank_level)
        for tile_id, tile_pe_cnt in twc_common_tile:
          if len(twc_pe_id) == len(iwc_pe_id):
            break
          if tile_id in bank_range:
            for pe_tuple in twc_tile_bank_id:
              if pe_tuple[1] == tile_id:
                twc_pe_id.append(pe_tuple[0])
                if len(twc_pe_id) == len(iwc_pe_id):
                  break
      assert (len(iwc_pe_id) == len(twc_pe_id))
      return iwc_pe_id, twc_pe_id

    # create new iwc and twc map
    self.topKCount = 0
    sort_iwc = np.flip(self.mapping.peInfo[self.mapping.peInfo[:, 1].argsort(), 0]).tolist()
    sort_twc = self.mapping.peInfo[self.mapping.peInfo[:, 2].argsort(), 0].tolist()
    swapped_pe = []
    if self.wlConfig.topKSize > 1:
      # save tiwl mapping
      if self.wlConfig.store_json_mapping:
        self.save_json_mapping(filepath=self.wlConfig.savePath + "_tiwl_%d.json" % self.stepCount,
                               xbar_mapping=self.generate_xbar_mapping(sort_iwc, sort_twc))

      PEBlock = getPEBlock()

      # topk_thres = (len(sort_iwc) / 2 ) * 0.3
      while len(sort_iwc) > 1:
        iwc_win = sort_iwc[:self.wlConfig.topKSize]
        twc_win = sort_twc[:self.wlConfig.topKSize]
        iwc_block_id = [(iwc, PEBlock[iwc]) for iwc in iwc_win]
        twc_tile_bank_id = [(twc, self.mapping.peTileBankID[twc][0], self.mapping.peTileBankID[twc][1]) for twc in twc_win]
        iwc_pe, twc_pe = swapSuitablePEs(iwc_block_id, twc_tile_bank_id)
        for i in range(len(iwc_pe)):
          pe_id_a = iwc_pe[i]
          pe_id_b = twc_pe[i]
          if pe_id_a not in swapped_pe and pe_id_b not in swapped_pe:
            if pe_id_a != pe_id_b:
              swapPE(pe_id_a, pe_id_b)
              sort_iwc.remove(pe_id_b)
              sort_twc.remove(pe_id_a)
              swapped_pe.append(pe_id_a)
              swapped_pe.append(pe_id_b)
            sort_iwc.remove(pe_id_a)
            sort_twc.remove(pe_id_b)
      # save top-k mapping
      if self.wlConfig.store_json_mapping:
        self.save_json_mapping(filepath=self.wlConfig.savePath + "_topk_%d.json" % self.stepCount,
                               xbar_mapping=self.get_xbar_mapping())
    else:
      # generate puma topk mapping json
      if self.wlConfig.store_json_mapping:
        sort_iwc_cp = sort_iwc.copy()
        sort_twc_cp = sort_twc.copy()
        src_pe_iwc = []
        dst_pe_twc = []

        PEBlock = getPEBlock()

        # topk_thres = (len(sort_iwc) / 2 ) * 0.3
        topk_size = int(self.mapping.peInfo.shape[0] / 2 * 0.8)

        while len(sort_iwc_cp) > 1:
          iwc_win = sort_iwc_cp[:topk_size]
          twc_win = sort_twc_cp[:topk_size]
          iwc_block_id = [(iwc, PEBlock[iwc]) for iwc in iwc_win]
          twc_tile_bank_id = [(twc, self.mapping.peTileBankID[twc][0], self.mapping.peTileBankID[twc][1]) for twc in twc_win]
          iwc_pe, twc_pe = swapSuitablePEs(iwc_block_id, twc_tile_bank_id)
          for i in range(len(iwc_pe)):
            pe_id_a = iwc_pe[i]
            pe_id_b = twc_pe[i]
            if pe_id_a not in swapped_pe and pe_id_b not in swapped_pe:
              if pe_id_a != pe_id_b:
                src_pe_iwc.append(pe_id_a)
                dst_pe_twc.append(pe_id_b)
                sort_iwc_cp.remove(pe_id_b)
                sort_twc_cp.remove(pe_id_a)
                swapped_pe.append(pe_id_a)
                swapped_pe.append(pe_id_b)
              sort_iwc_cp.remove(pe_id_a)
              sort_twc_cp.remove(pe_id_b)
        # save top-k mapping
        assert len(src_pe_iwc) == len(dst_pe_twc)
        src_pe_iwc_cp = src_pe_iwc.copy()
        dst_pe_twc_cp = dst_pe_twc.copy()
        src_pe_iwc = src_pe_iwc + dst_pe_twc_cp
        dst_pe_twc = dst_pe_twc + src_pe_iwc_cp
        self.save_json_mapping(filepath=self.wlConfig.savePath + "_topk_%d.json" % self.stepCount,
                               xbar_mapping=self.generate_xbar_mapping(src_pe_iwc, dst_pe_twc))

      # =========================
      while len(sort_iwc) > 1:
        pe_id_a = sort_iwc[0]
        pe_id_b = sort_twc[0]
        if pe_id_a != pe_id_b:
          swapPE(pe_id_a, pe_id_b)
          sort_iwc.remove(pe_id_b)
          sort_twc.remove(pe_id_a)
        sort_iwc.remove(pe_id_a)
        sort_twc.remove(pe_id_b)

      # save tiwl mapping
      if self.wlConfig.store_json_mapping:
        self.save_json_mapping(filepath=self.wlConfig.savePath + "_tiwl_%d.json" % self.stepCount,
                               xbar_mapping=self.get_xbar_mapping())

  def intraArrayShift(self, isFinalWL):
    """
    Shifts crossbar within a PE.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    for pe_id in self.mapping.peid2pid:
      new_pid_list = np.roll(self.mapping.peid2pid[pe_id], shift=(1), axis=(0))

      # backup the final one of old_pid_list
      last_logical_key_old = self.mapping.pid2lid[self.mapping.peid2pid[pe_id][-1]]
      pidMap2D_idx = np.where(self.mapping.logicalArrayDict[last_logical_key_old].pidMap2D == self.mapping.peid2pid[pe_id][-1])
      cellCurrentTensor = self.mapping.cellCurrentTensorDict[self.mapping.peid2pid[pe_id][-1]]

      for pid_idx in range(len(new_pid_list) - 1):
        # update self.pid2lidDataFrame
        old_logical_key = self.mapping.pid2lid[self.mapping.peid2pid[pe_id][pid_idx]]
        old_idx = np.where(self.mapping.logicalArrayDict[old_logical_key].pidMap2D == self.mapping.peid2pid[pe_id][pid_idx])

        # update self.logicalArrayDict
        self.mapping.pid2lid[new_pid_list[pid_idx]] = old_logical_key
        self.mapping.logicalArrayDict[old_logical_key].pidMap2D[old_idx] = new_pid_list[pid_idx]

        if not self.wlConfig.overlapUpdate:
          diff = self.mapping.cellCurrentTensorDict[new_pid_list[pid_idx]] ^ self.mapping.cellCurrentTensorDict[self.mapping.peid2pid[pe_id][pid_idx]]
          self.updateWriteCountByDifference(new_pid_list[pid_idx], diff, isFinalWL)
          self.updateCellCurrentTensor(new_pid_list[pid_idx], self.mapping.cellCurrentTensorDict[self.mapping.peid2pid[pe_id][pid_idx]])

      self.mapping.pid2lid[new_pid_list[-1]] = last_logical_key_old
      self.mapping.logicalArrayDict[last_logical_key_old].pidMap2D[pidMap2D_idx] = new_pid_list[-1]

      if not self.wlConfig.overlapUpdate:
        diff = self.mapping.cellCurrentTensorDict[new_pid_list[-1]] ^ cellCurrentTensor
        self.updateWriteCountByDifference(new_pid_list[-1], diff, isFinalWL)
        self.updateCellCurrentTensor(new_pid_list[-1], cellCurrentTensor)


  def updateWriteCountDict(self, pid, writeCnt):
    """
    Updates cell write count dict of wear-leveling manager.
    @param pid: PID of the physical array.
    @param writeCnt: cell writes count tensor of the physical array.
    @return:
    """
    if writeCnt.size(dim=0) < self.mapping.cellWriteCountDict[pid].size(dim=0) or writeCnt.size(dim=1) < \
        self.mapping.cellWriteCountDict[pid].size(dim=1):
      p2d = (0, self.mapping.cellWriteCountDict[pid].size(dim=1) - writeCnt.size(dim=1), 0, self.mapping.cellWriteCountDict[pid].size(dim=0) - writeCnt.size(dim=0))
      writeCnt = F.pad(writeCnt, pad=p2d, mode='constant', value=0)

    pe_id = self.mapping.physicalArrayInfo[pid][1]

    self.mapping.cellWriteCountDict[pid] += writeCnt
    writeCntSum = writeCnt.sum().item()

    self.mapping.physicalArrayInfo[pid][2:4] += writeCntSum
    self.mapping.peInfo[pe_id][1:3] += writeCntSum


  def updateCellCurrentTensor(self, pid, cellDataTensor):
    """
    Updates cell current tensor dict for wear-leveling manager.
    @param pid: PID of physical array.
    @param cellDataTensor: cell current tensor.
    @return:
    """
    if cellDataTensor.shape[0] < archConst.phyArraySize[0] or cellDataTensor.shape[1] < archConst.phyArraySize[1]:
      p2d = (0, archConst.phyArraySize[1] - cellDataTensor.size(dim=1), 0, archConst.phyArraySize[0] - cellDataTensor.size(dim=0))
      self.mapping.cellCurrentTensorDict[pid] = F.pad(cellDataTensor, pad=p2d, mode='constant', value=0)
    else:
      self.mapping.cellCurrentTensorDict[pid] = cellDataTensor

  def updateWriteCountByDifference(self, pid, diff, isFinalWL):
    """
    Updates writes count dict, IWC and TWC of specific physical array.
    @param pid: PID of physical array.
    @param diff: Difference tensor
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    if diff.sum() != 0:
      for i in range(archConst.cellBits):
        wrtCnt = diff & 1
        diff.bitwise_right_shift_(1)
        self.mapping.cellWriteCountDict[pid] += wrtCnt
        wrtCntSum = wrtCnt.sum()
        if isFinalWL:
          pid_idx = np.where(self.mapping.physicalArrayInfo[:, 0] == pid)[0][0]
          pe_id = self.mapping.physicalArrayInfo[pid_idx][1]
          pe_id_idx = np.where(self.mapping.peInfo[:, 0] == pe_id)[0][0]
          self.mapping.physicalArrayInfo[pid_idx][3] += wrtCntSum
          self.mapping.peInfo[pe_id_idx][2] += wrtCntSum


  def get_pe_mapping(self):
    pe_mapping = {}
    pe_shift_bits = self.arch._mask_size_list[ArchLevel.pe_level - 1]
    for array_name, array in self.mapping.logicalArrayDict.items():
      pe_mapping[array_name] = []
      for row_idx, row_pid in enumerate(array.pidMap2D):
        row_pe = np.array([pid >> pe_shift_bits for pid in row_pid])
        pe_mapping[array_name].append(np.unique(row_pe))
      pe_mapping[array_name] = np.array(pe_mapping[array_name]).transpose()
    return pe_mapping

  def generate_pe_mapping(self, src_pe_list, dst_pe_list):
    pe_mapping = {}
    pe_shift_bits = self.arch._mask_size_list[ArchLevel.pe_level - 1]
    for array_name, array in self.mapping.logicalArrayDict.items():
      pe_mapping[array_name] = []
      for row_idx, row_pid in enumerate(array.pidMap2D):
        row_pe = np.array([pid >> pe_shift_bits for pid in row_pid])
        row_pe = np.unique(row_pe)
        for i, pe in enumerate(row_pe):
          if pe in src_pe_list:
            row_pe[i] = dst_pe_list[src_pe_list.index(pe)]
        pe_mapping[array_name].append(row_pe)
      pe_mapping[array_name] = np.array(pe_mapping[array_name]).transpose()
    return pe_mapping


  def get_xbar_mapping(self):
    pid2peid = {}
    for peid, pid_list in self.mapping.peid2pid.items():
      for pid in pid_list:
        pid2peid[pid] = peid

    xbar_mapping = {}
    for array_name, array in self.mapping.logicalArrayDict.items():
      xbar_mapping[array_name] = np.zeros_like(array.pidMap2D)
      for idx, pid in np.ndenumerate(array.pidMap2D):
        src_pe = pid2peid[pid]
        interval_idx = self.mapping.peid2pid[src_pe].index(pid)
        xbar_mapping[array_name][idx] = self.mapping.peid2pid[src_pe][interval_idx]
    return xbar_mapping


  def generate_xbar_mapping(self, src_pe_list, dst_pe_list):
    pid2peid = {}
    for peid, pid_list in self.mapping.peid2pid.items():
      for pid in pid_list:
        pid2peid[pid] = peid

    xbar_mapping = {}
    for array_name, array in self.mapping.logicalArrayDict.items():
      xbar_mapping[array_name] = np.zeros_like(array.pidMap2D)
      for idx, pid in np.ndenumerate(array.pidMap2D):
        src_pe = pid2peid[pid]
        interval_idx = self.mapping.peid2pid[src_pe].index(pid)
        if src_pe in src_pe_list:
          dst_pe = dst_pe_list[src_pe_list.index(src_pe)]
          xbar_mapping[array_name][idx] = self.mapping.peid2pid[dst_pe][interval_idx]
        else:
          xbar_mapping[array_name][idx] = self.mapping.peid2pid[src_pe][interval_idx]
    return xbar_mapping

  def save_json_mapping(self, filepath, xbar_mapping):
    class NumpyArrayEncoder(JSONEncoder):
      def default(self, obj):
        if isinstance(obj, np.ndarray):
          return obj.tolist()
        return JSONEncoder.default(self, obj)

    # convert xbar mapping to puma mapping
    vmvmu_mapping = {}
    vmvmu_list = []
    cellNumPerValue = math.ceil(16 / archConst.cellBits)

    for layer_name, mapping in xbar_mapping.items():
      if not layer_name.startswith("Placeholder-"):
        mapping_reshape = mapping.reshape(mapping.shape[1], mapping.shape[0])
        for sub_layer_idx in range(cellNumPerValue):
          sub_layer_name = layer_name + ".%d" % sub_layer_idx
          col_len = mapping.shape[1] // cellNumPerValue
          sub_layer_mapping = mapping_reshape[sub_layer_idx * col_len: (sub_layer_idx + 1) * col_len, :]
          vmvmu_mapping[sub_layer_name] = sub_layer_mapping
          vmvmu_list += vmvmu_mapping[sub_layer_name].flatten().tolist()
      else:
        vmvmu_mapping["placeholder"] = mapping.transpose()
        vmvmu_list += vmvmu_mapping["placeholder"].flatten().tolist()
    vmvmu_list.sort()
    for i in range(len(vmvmu_list)):
      assert vmvmu_list[i] == i

    # PUMA mapping should be transposed
    with open(filepath.replace(".pkl", ".json"), "w") as jsonFile:
      json.dump(vmvmu_mapping, jsonFile, cls=NumpyArrayEncoder)

  def save_wl_data(self, filepath, save_space=True):
    batch_idx = self.stepCount - 1
    liteCellWriteCountDict = {}
    if not save_space:
      for pid in self.mapping.cellWriteCountDict:
        liteCellWriteCountDict[pid] = self.mapping.cellWriteCountDict[pid].to(torch.int32)
    else:
      for pid in self.mapping.cellWriteCountDict:
        key_points = []
        cellWrtCnt = self.mapping.cellWriteCountDict[pid].flatten().numpy()
        for pos in [0.07, 0.16, 0.31, 0.5, 0.69, 0.84, 0.93]:
          k = int(len(cellWrtCnt) * pos)
          key_points.append(cellWrtCnt[np.argpartition(cellWrtCnt, k)][:k].max())
        key_points.insert(0, cellWrtCnt.min())
        key_points.append(cellWrtCnt.max())
        liteCellWriteCountDict[pid] = {
          "mean": cellWrtCnt.mean(),
          "key_points": key_points
        }

    wl_data = {
      "step": batch_idx,
      "pid2lid": self.mapping.pid2lid,
      "peid2pid": self.mapping.peid2pid,
      "physicalArrayInfo": self.mapping.physicalArrayInfo,
      "peInfo": self.mapping.peInfo,
      "cellWriteCountDict": liteCellWriteCountDict,
      "mappingState": self.mappingState,
      "comm_data": self.comm_data,
      "mapping": self.get_pe_mapping()
    }

    with open(filepath, "wb") as f:
      torch.save(wl_data, f)


# Utils
def copy_fixed_point_params(nn_model):
  params_dict = {}
  for name, layer in nn_model.named_modules():
    if hasattr(layer, 'weightBits') and hasattr(layer, 'fp_weight'):
      params_dict[name] = fpA.to_int(layer.fp_weight).clone()
  return params_dict


