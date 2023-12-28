import json
import os
import warnings
import uuid
import torch
import numpy as np
from enum import Enum
from collections import Counter
from json import JSONEncoder

from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.addressParser import ArchLevel, ArchID
from src.pimtorch.simulator.resourceManager import ResourceManager
from src.pimtorch.simulator.mapping.mappingTable import MappingTable
from src.pimtorch.simulator.mapping.logicalTensor import LogicalTensorForWL
from src.pimtorch.simulator.mapping.allocator import AlignAllocator
from src.pimtorch.simulator.extension.dynamicPerformance.communication import CommAnalyzer
import src.pimtorch.nn.fixedPointArithmetic as fpA

class CrossbarWlType(Enum):
  ColShift = "ColShift"
  RowShift = "RowShift"
  RowColShift = "RowColShift"
  ZShift = "ZShift"

class FillingType(Enum):
  Multiple = "Multiple"
  All = "All"
  NoFill = "NoFill"

class WearLeveling:
  def __init__(self, resourceManager: ResourceManager, mappingTable: MappingTable):
    self.jobName = cfg.DataName
    self.stepCount = 0
    self.zShiftStepCount = 0
    self.resManager = resourceManager
    self.mappingTable = mappingTable

    # init parameters
    self.TIWLInterval = cfg.TIWLInterval
    self.PEShiftInterval = cfg.PEShiftInterval
    self.crxWlInterval = cfg.crxWlInterval
    self.crxWlType = CrossbarWlType(cfg.crxWlType)
    self.LASRTopK = cfg.LASRTopK
    self.granularity = getattr(ArchLevel, cfg.granularityTIWL)
    self.overlapUpdate = cfg.overlapUpdate
    self.fillingType = FillingType(cfg.fillingType)
    self.fillingParam = cfg.fillingParam
    self.shiftNum = cfg.shiftNum
    self.storeTranspose = cfg.storeTranspose
    self.saveInterval = cfg.saveInterval
    self.saveDir = cfg.saveDir
    self.saveLiteWlState = cfg.saveLiteWlState
    self.saveJsonMapping = cfg.saveJsonMapping

    self.useTIWL = self.TIWLInterval > 0
    self.usePEShift = self.granularity == ArchLevel.pe and self.PEShiftInterval > 0
    self.useCrxShift = self.crxWlInterval > 0
    self.useLASR = self.LASRTopK > 0

    # init global write counts, record granularity is the crossbar
    self.writeCounts = torch.zeros(size=(*cfg.archInfo[:-1], *cfg.crxShape), dtype=cfg.torchInt, requires_grad=False)
    self.lastWriteCounts = torch.zeros_like(self.writeCounts, dtype=cfg.torchInt, requires_grad=False)
    self.lastCellBit = torch.zeros_like(self.writeCounts, dtype=cfg.torchInt, requires_grad=False)

    assert self.LASRTopK <= 1
    assert self.granularity == ArchLevel.crx or self.granularity == ArchLevel.pe, \
      "The granularity of TIWL should be crossbar or PE."
    if self.useLASR:
      assert self.granularity == ArchLevel.pe, "LASR only support scheduling at PE granularity."


  def initByModule(self, dnnModule: torch.nn.Module, fillingType: FillingType, analyzeComm=False):
    """
    Init wear-leveling manager by given pytorch nn module.
    @param dnnModule: pytorch nn module
    @return:
    """
    if not self.overlapUpdate:
      warnings.warn("You're training DNN module, please turn on overlapUpdate.")
      self.overlapUpdate = True
    for name, layer in dnnModule.named_modules():
      if hasattr(layer, 'fp_weight'):
        sz = layer.fp_weight.size()
        self.mappingTable.addLogicalTensor(
          LogicalTensorForWL(size=sz, name=name, resourceManager=self.resManager, mappingTable=self.mappingTable,
                             allocator=AlignAllocator, wearLevelingManager=self)
        )
        # store transpose of weight
        if self.storeTranspose:
          self.mappingTable.addLogicalTensor(
            LogicalTensorForWL(size=(sz[1], sz[0]), name=name+".t", resourceManager=self.resManager,
                               mappingTable=self.mappingTable, allocator=AlignAllocator, wearLevelingManager=self)
          )

    # write initial weights
    for name, layer in dnnModule.named_modules():
      if hasattr(layer, 'weightBits') and hasattr(layer, 'fp_weight'):
        self.mappingTable.getLogicalTensor(tensorName=name).write(fpA.to_int(layer.fp_weight).clone().detach(), False, False)
        if self.storeTranspose:
          self.mappingTable.getLogicalTensor(tensorName=name+".t").write(fpA.to_int(layer.fp_weight).t().clone().detach(), False, False)
    self.__filling__(fillingType)
    self.crxIDAndIdx = self.resManager.getUsedArchLevelIDAndIdx(ArchLevel.crx)
    self.peIDAndIdx = self.resManager.getUsedArchLevelIDAndIdx(ArchLevel.pe)
    if self.granularity == ArchLevel.crx:
      self.winSizeOfLASR = int(len(self.crxIDAndIdx) * self.LASRTopK)
    else:
      self.winSizeOfLASR = int(len(self.peIDAndIdx) * self.LASRTopK)
    self.analyzeComm = analyzeComm
    if self.analyzeComm:
      self.commAnalyzer = CommAnalyzer(dnnModule, self.mappingTable)
      self.commAnalyzer.analysis()

  def __filling__(self, fillingType: FillingType):
    if fillingType == FillingType.Multiple:
      used_n = self.resManager.getUsedNum(ArchLevel.crx)
      alloc_n = int(used_n * self.fillingParam) - used_n
      assert alloc_n > 0 and (used_n + alloc_n) <= self.resManager.getNum(ArchLevel.crx)
    elif fillingType == FillingType.All:
      alloc_n = self.resManager.getFreeNum(ArchLevel.crx)
      assert alloc_n > 0
    elif fillingType == FillingType.NoFill:
      return
    else:
      raise NotImplementedError("This filling mode is not supported.")
    if cfg.dataSplitting:
      size = (alloc_n // cfg.cellNumPerValue) * cfg.crxShape[1]
    else:
      size = (alloc_n * cfg.crxShape[1]) // cfg.cellNumPerValue
    self.placeholderName = "Placeholder_" + str(uuid.uuid4())
    self.mappingTable.addLogicalTensor(
      LogicalTensorForWL(size=(1, size), name=self.placeholderName, resourceManager=self.resManager,
                         mappingTable=self.mappingTable, allocator=AlignAllocator, wearLevelingManager=self)
    )

  def initByTensorList(self, tensorList):
    """
    Init wear-leveling manager by given tensors list.
    @param tensorList: pytorch tensors list
    @param bitWidth:
    @return:
    """
    names = []
    for idx, tensor in enumerate(tensorList):
      name = str(uuid.uuid4())
      self.mappingTable.addLogicalTensor(
        LogicalTensorForWL(size=tensor.shape, name=name, resourceManager=self.resManager, mappingTable=self.mappingTable,
                           allocator=AlignAllocator, wearLevelingManager=self)
      )
      names.append(name)
    self.crxIDAndIdx = self.resManager.getUsedArchLevelIDAndIdx(ArchLevel.crx)
    self.peIDAndIdx = self.resManager.getUsedArchLevelIDAndIdx(ArchLevel.pe)
    if self.granularity == ArchLevel.crx:
      self.winSizeOfLASR = int(len(self.crxIDAndIdx) * self.LASRTopK)
    else:
      self.winSizeOfLASR = int(len(self.peIDAndIdx) * self.LASRTopK)
    return names

  def updateWriteCounts(self, crxID: int, writeCounts: torch.Tensor):
    npIdx = ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx)
    self.writeCounts[npIdx] += writeCounts

  def updateLastCellBit(self, crxID: int, cellBit: torch.Tensor):
    npIdx = ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx)
    self.lastCellBit[npIdx] = cellBit

  def step(self, data_dict: dict):
    """
    Performs a single write step. The wear-leveling operation would be performed at wear-leveling
    intervals.
    @param old_data_dict: old matrix datas
    @param new_data_dict: new matrix datas
    @return:
    """
    # flush write counts or update last cell bit
    if self.useTIWL and self.stepCount % self.TIWLInterval == 0:
      flushWrtCnt = True
    elif self.usePEShift and self.stepCount % self.PEShiftInterval == 0:
      flushWrtCnt = True
    elif self.useCrxShift and self.stepCount % self.crxWlInterval == 0:
      flushWrtCnt = True
    else:
      flushWrtCnt = False

    if self.useTIWL and self.stepCount % (self.TIWLInterval - 1) == 0:
      updateLastCellBit = True
    elif self.usePEShift and self.stepCount % (self.PEShiftInterval - 1) == 0:
      updateLastCellBit = True
    elif self.useCrxShift and self.stepCount % (self.crxWlInterval - 1) == 0:
      updateLastCellBit = True
    else:
      updateLastCellBit = False

    for name in data_dict:
      self.mappingTable.getLogicalTensor(tensorName=name).write(data_dict[name],
                                                                flushWriteCounts=flushWrtCnt,
                                                                updateLastCellBit=updateLastCellBit)
      if self.storeTranspose:
        self.mappingTable.getLogicalTensor(tensorName=name+".t").write(data_dict[name].t(),
                                                                       flushWriteCounts=flushWrtCnt,
                                                                       updateLastCellBit=updateLastCellBit)

    if self.stepCount % self.crxWlInterval == 0:
      if self.crxWlType != CrossbarWlType.ZShift:
        shifts = (0, 0)
        if self.crxWlType == CrossbarWlType.RowColShift:
          shifts = (1, 1)
        elif self.crxWlType == CrossbarWlType.RowShift:
          shifts = (1, 0)
        elif self.crxWlType == CrossbarWlType.ColShift:
          shifts = (0, 1)
      else:
        shifts = (0, 1)
        if self.zShiftStepCount % cfg.crxShape[1] == 0:
          shifts = (1, 1)
        self.zShiftStepCount += 1
      self.shifting(shifts=shifts)
    if self.stepCount % self.PEShiftInterval == 0:
      self.PECyclicShift()
    if self.stepCount % self.TIWLInterval == 0:
      self.TIWL()
    if self.stepCount % self.saveInterval == 0:
      prefix = "%s_%s" % (self.jobName, cfg.timeStamp.strftime("%Y-%m-%d-%H:%M:%S"))
      self.saveWlState(filepath=os.path.join(self.saveDir, prefix + ".pkl"), save_space=self.saveLiteWlState)
      if self.saveJsonMapping:
        self.saveMapping2D(filepath=os.path.join(self.saveDir, prefix + ".json"))
    self.stepCount += 1


  def shifting(self, shifts: tuple):
    """
    Performs shifting intra-crossbar wear-leveling scheme.
    @param isFinalWL: If this is the final wera-leveling operation, update the IWC and TWC.
    @return:
    """
    assert len(shifts) == 2, "The shifting() function expects a tuple of length 2."
    if not self.overlapUpdate:
      shiftedWrtCnt = torch.roll(self.lastCellBit, shifts=shifts, dims=(-2, -1))
      self.writeCounts += self.lastCellBit ^ shiftedWrtCnt
    self.writeCounts = torch.roll(self.writeCounts, shifts=shifts, dims=(-2, -1))

  def __swapArchLevel__(self, archLevelID_a, archLevelID_b):
    def updateWriteCounts():
      levelIdx_a = ArchID.archLevelID2archLevelIdx(archLevelID_a, self.granularity)
      levelIdx_b = ArchID.archLevelID2archLevelIdx(archLevelID_b, self.granularity)
      diff = self.lastCellBit[levelIdx_a] ^ self.lastCellBit[levelIdx_b]
      self.writeCounts[levelIdx_a] += diff
      self.writeCounts[levelIdx_b] += diff

    def swapNumpyArray(arr, lNpIdx, rNpIdx):
      if isinstance(arr, np.ndarray):
        cp = arr[lNpIdx].copy()
        arr[lNpIdx] = arr[rNpIdx]
        arr[rNpIdx] = cp
      else:
        cp = arr[lNpIdx].clone().detach()
        arr[lNpIdx] = arr[rNpIdx]
        arr[rNpIdx] = cp

    def swapMapping2D(crxIDs_a, crxIDs_b):
      crxIdx_a = [(*ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx), 0) for crxID in crxIDs_a]
      crxIdx_b = [(*ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx), 0) for crxID in crxIDs_b]
      for i, (idx_a, idx_b) in enumerate(zip(crxIdx_a, crxIdx_b)):
        tensor_a = self.mappingTable.getLogicalTensor(archLevelIdx=idx_a)
        tensor_b = self.mappingTable.getLogicalTensor(archLevelIdx=idx_b)
        if tensor_a:
          if tensor_b:
            value_a = tensor_b.mapping2D[tuple(self.mappingTable.globalIdx2mappingIdx[idx_b])]
          else:
            value_a = crxIDs_b[i]
          tensor_a.mapping2D[tuple(self.mappingTable.globalIdx2mappingIdx[idx_a])] = value_a
        if tensor_b:
          if tensor_a:
            value_b = tensor_a.mapping2D[tuple(self.mappingTable.globalIdx2mappingIdx[idx_a])]
          else:
            value_b = crxIDs_a[i]
          tensor_b.mapping2D[tuple(self.mappingTable.globalIdx2mappingIdx[idx_b])] = value_b


    crxIDs_a = self.resManager.getArchLevelIDs(archLevelID_a, self.granularity, ArchLevel.crx)
    crxIDs_b = self.resManager.getArchLevelIDs(archLevelID_b, self.granularity, ArchLevel.crx)
    if not self.overlapUpdate:
      updateWriteCounts()
    swapMapping2D(crxIDs_a, crxIDs_b)
    lNpIdx = ArchID.archLevelID2archLevelIdx(archLevelID_a, self.granularity)
    rNpIdx = ArchID.archLevelID2archLevelIdx(archLevelID_b, self.granularity)
    swapNumpyArray(self.mappingTable.globalIdx2mappingIdx, lNpIdx, rNpIdx)
    swapNumpyArray(self.mappingTable.globalIdx2TensorID, lNpIdx, rNpIdx)
    # swapNumpyArray(self.writeCounts, lNpIdx, rNpIdx)

  def __PECyclicShift__(self, peID):
    def updateWriteCounts(crxIDs):
      crxIdx = [ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx) for crxID in crxIDs]
      for i in range(len(crxIdx)):
        diff = self.lastCellBit[crxIdx[i]] ^ self.lastCellBit[crxIdx[i-1]]
        self.writeCounts[crxIdx[i]] += diff

    def cyclicShiftNumpyArray(arr, npIdx):
      old_indice = np.arange(cfg.crx_n)
      new_indice = np.roll(old_indice, -1)
      arr[npIdx][old_indice] = arr[npIdx][new_indice]

    def updateMapping2D(crxIDs):
      newCrxIDs = np.roll(crxIDs, shift=1)
      globalIdx = [(*ArchID.archLevelID2archLevelIdx(crxID, ArchLevel.crx), 0) for crxID in crxIDs]
      for i, idx in enumerate(globalIdx):
        tensorID = self.mappingTable.globalIdx2TensorID[idx]
        if tensorID != 0:
          mappingIdx = tuple(self.mappingTable.globalIdx2mappingIdx[idx])
          self.mappingTable.logicalTensors[tensorID].mapping2D[mappingIdx] = newCrxIDs[i]


    crxIDs = self.resManager.getArchLevelIDs(peID, ArchLevel.pe, ArchLevel.crx)
    if not self.overlapUpdate:
      updateWriteCounts(crxIDs)
    updateMapping2D(crxIDs)
    npIdx = ArchID.archLevelID2archLevelIdx(peID, ArchLevel.pe)
    cyclicShiftNumpyArray(self.mappingTable.globalIdx2mappingIdx, npIdx)
    cyclicShiftNumpyArray(self.mappingTable.globalIdx2TensorID, npIdx)

  def TIWL(self):
    class WrtCntPair:
      def __init__(self, npIdx: tuple, writeCounts: int):
        self.npIdx = npIdx
        self.wrtCnt = writeCounts
        self.peID, _ = ArchID.archLevelIdx2archLevelID(self.npIdx)
        self.tileID = self.peID // cfg.pe_n
        self.bankID = self.tileID // cfg.tile_n

      def __lt__(self, other):
        return self.wrtCnt < other.wrtCnt
      def __gt__(self, other):
        return self.wrtCnt > other.wrtCnt
      def __eq__(self, other):
        return self.npIdx == other.npIdx
      def __str__(self):
        return "npIdx: %s, wrtCnt: %d" % (self.npIdx, self.wrtCnt)
      def __repr__(self):
        return "npIdx: %s, wrtCnt: %d" % (self.npIdx, self.wrtCnt)

    def swap(wrtCntPair_a, wrtCntPair_b):
      if self.granularity == ArchLevel.pe:
        self.__swapArchLevel__(wrtCntPair_a.peID, wrtCntPair_b.peID)
      else:
        self.__swapArchLevel__(wrtCntPair_a.crxID, wrtCntPair_b.crxID)

    def buildWrtCntTable():
      sorted_iwc = []
      sorted_twc = []
      iwc = self.writeCounts - self.lastWriteCounts
      if self.granularity == ArchLevel.crx:
        iwc_sum = iwc.sum(dim=(-2, -1))
        twc_sum = self.writeCounts.sum(dim=(-2, -1))
        for crxID, npIdx in self.crxIDAndIdx:
          sorted_iwc.append(WrtCntPair(npIdx, iwc_sum[npIdx]))
          sorted_twc.append(WrtCntPair(npIdx, twc_sum[npIdx]))
      else:
        iwc_sum = iwc.sum(dim=(-3, -2, -1))
        twc_sum = self.writeCounts.sum(dim=(-3, -2, -1))
        for peID, npIdx in self.peIDAndIdx:
          sorted_iwc.append(WrtCntPair(npIdx, iwc_sum[npIdx]))
          sorted_twc.append(WrtCntPair(npIdx, twc_sum[npIdx]))
      sorted_iwc.sort(reverse=True)
      sorted_twc.sort()
      return sorted_iwc, sorted_twc

    def getPEBlock():
      PEBlock = {}
      for _, logicalTensor in self.mappingTable.logicalTensors.items():
        logicalTensor.updatePEBlock()
      for peID, npIdx in self.peIDAndIdx:
        tensorIDs = np.unique(self.mappingTable.globalIdx2TensorID[npIdx])
        tensorID = tensorIDs[0] if tensorIDs[0] else tensorIDs[1]
        if peID not in PEBlock:
          PEBlock[peID] = self.mappingTable.logicalTensors[tensorID].PEBlock[peID]
      return PEBlock

    def LASR(iwc_block_id, twc_tile_bank_id):
      iwc_unzip = list(zip(*iwc_block_id))
      twc_unzip = list(zip(*twc_tile_bank_id))
      iwc_common_counter = Counter(iwc_unzip[1]).most_common()
      twc_common_tile = Counter(twc_unzip[1]).most_common()
      twc_common_bank = Counter(twc_unzip[2]).most_common()
      iwc_wrtCntPair = []
      twc_wrtCntPair = []
      # Placeholder should be processed at the end.
      if len(iwc_common_counter) > 1 and ("Null" in iwc_common_counter[0][0] or "Placeholder" in iwc_common_counter[0][0]):
        iwc_common_stripe = iwc_common_counter[1]
      else:
        iwc_common_stripe = iwc_common_counter[0]
      for pe_tuple in iwc_block_id:
        if len(iwc_wrtCntPair) < iwc_common_stripe[1]:
          if pe_tuple[1] == iwc_common_stripe[0]:
            iwc_wrtCntPair.append(pe_tuple[-1])
        else:
          break
      for bank_id, bank_pe_cnt in twc_common_bank:
        if len(twc_wrtCntPair) == len(iwc_wrtCntPair):
          break
        bank_range = self.resManager.getArchLevelIDs(bank_id, ArchLevel.bank, ArchLevel.tile)
        for tile_id, tile_pe_cnt in twc_common_tile:
          if len(twc_wrtCntPair) == len(iwc_wrtCntPair):
            break
          if tile_id in bank_range:
            for pe_tuple in twc_tile_bank_id:
              if pe_tuple[1] == tile_id:
                twc_wrtCntPair.append(pe_tuple[-1])
                if len(twc_wrtCntPair) == len(iwc_wrtCntPair):
                  break
      selected_pe = []
      iwc_ptr = 0
      twc_ptr = 0
      while iwc_ptr < len(iwc_wrtCntPair) and twc_ptr < len(twc_wrtCntPair):
        target_a = iwc_wrtCntPair[iwc_ptr]
        target_b = twc_wrtCntPair[twc_ptr]
        if target_a not in selected_pe and target_b not in selected_pe:
          selected_pe += [target_a, target_b]
          if target_a != target_b:
            if target_b in iwc_wrtCntPair:
              iwc_wrtCntPair.remove(target_b)
            if target_a in twc_wrtCntPair:
              twc_wrtCntPair.remove(target_a)
          iwc_ptr += 1
          twc_ptr += 1
        else:
          if target_a in selected_pe:
            iwc_wrtCntPair.remove(target_a)
          if target_b in selected_pe:
            twc_wrtCntPair.remove(target_b)
      iwc_wrtCntPair = iwc_wrtCntPair[:iwc_ptr]
      twc_wrtCntPair = twc_wrtCntPair[:twc_ptr]
      assert (len(iwc_wrtCntPair) == len(twc_wrtCntPair))
      return iwc_wrtCntPair, twc_wrtCntPair


    # (npIdx: (chip, bank, tile, pe, crx), wrtCnt)
    sorted_iwc, sorted_twc = buildWrtCntTable()
    if self.useLASR:
      PEBlock = getPEBlock()
      swapped_pe = []
      while len(sorted_iwc) > 0:
        iwc_win = [(pair.peID, PEBlock[pair.peID], pair) for pair in sorted_iwc[:self.winSizeOfLASR]]
        twc_win = [(pair.peID, pair.tileID, pair.bankID, pair) for pair in sorted_twc[:self.winSizeOfLASR]]
        selected_iwc_pe, selected_twc_pe = LASR(iwc_win, twc_win)
        for target_a, target_b in zip(selected_iwc_pe, selected_twc_pe):
          if target_a != target_b:
            if target_a not in swapped_pe and target_b not in swapped_pe:
              swap(target_a, target_b)
              swapped_pe += [target_a, target_b]
            if target_b in sorted_iwc:
              sorted_iwc.remove(target_b)
            if target_a in sorted_twc:
              sorted_twc.remove(target_a)
          else:
            swapped_pe.append(target_a)
          if target_a in sorted_iwc:
            sorted_iwc.remove(target_a)
          if target_b in sorted_twc:
            sorted_twc.remove(target_b)
    else:
      while len(sorted_iwc) > 1:
        target_a = sorted_iwc[0]
        target_b = sorted_twc[0]
        if target_a != target_b:
          swap(target_a, target_b)
          sorted_iwc.remove(target_b)
          sorted_twc.remove(target_a)
        sorted_iwc.remove(target_a)
        sorted_twc.remove(target_b)

    # update last write counts
    self.lastWriteCounts = self.writeCounts.clone().detach()

  def PECyclicShift(self):
    peIDs = set([ArchID.archLevelIdx2archLevelID(levelIdx[:-2])
                 for levelIdx in zip(*np.nonzero(self.mappingTable.globalIdx2TensorID))])
    for peID, _ in peIDs:
      self.__PECyclicShift__(peID=peID)

  def saveMapping2D(self, filepath):
    class NumpyArrayEncoder(JSONEncoder):
      def default(self, obj):
        if isinstance(obj, np.ndarray):
          return obj.tolist()
        return JSONEncoder.default(self, obj)

    # convert xbar mapping to puma mapping
    vmvmu_mapping = {}
    vmvmu_list = []
    for tensorID, tensor in self.mappingTable.logicalTensors.items():
      if not tensor.name.startswith("Placeholder-"):
        mapping_t = tensor.mapping2D.transpose()
        for sub_layer_idx in range(cfg.cellNumPerValue):
          sub_layer_name = tensor.name + ".%d" % sub_layer_idx
          col_len = tensor.mapping2D.shape[1] // cfg.cellNumPerValue
          sub_layer_mapping = mapping_t[sub_layer_idx * col_len: (sub_layer_idx + 1) * col_len, :]
          vmvmu_mapping[sub_layer_name] = sub_layer_mapping
          vmvmu_list += vmvmu_mapping[sub_layer_name].flatten().tolist()
      else:
        vmvmu_mapping["placeholder"] = tensor.mapping2D.transpose()
        vmvmu_list += vmvmu_mapping["placeholder"].flatten().tolist()
    vmvmu_list.sort()
    for i in range(len(vmvmu_list)):
      assert vmvmu_list[i] == i

    # PUMA mapping should be transposed
    with open(filepath, "w") as jsonFile:
      json.dump(vmvmu_mapping, jsonFile, cls=NumpyArrayEncoder)

  def saveWlState(self, filepath, save_space=True):
    batch_idx = self.stepCount - 1
    wrtCntFlatten = self.writeCounts.view(*self.writeCounts.shape[:-2], -1)
    cellNum = cfg.crxShape[0] * cfg.crxShape[1] - 1
    posIdx = [0, 0.07, 0.16, 0.31, 0.5, 0.69, 0.84, 0.93, 1]
    posIdx = [int(i * cellNum) for i in posIdx]
    argIndice = torch.argsort(wrtCntFlatten)

    if save_space:
      wrtCount = {}
      for crxID, crxIdx in self.crxIDAndIdx:
        key_points = []
        for pos in posIdx:
          key_points.append(wrtCntFlatten[crxIdx][argIndice[crxIdx][pos]].item())
        wrtCount[crxID] = {
          "mean": wrtCntFlatten[crxIdx].float().mean().item(),
          "key_points": key_points
        }
    else:
      wrtCount = self.writeCounts

    tensorID2mapping2D = {}
    for tensorID, tensor in self.mappingTable.logicalTensors.items():
      tensorID2mapping2D[tensorID] = tensor.mapping2D

    wl_data = {
      "step": batch_idx,
      "mappingTable": {
        "name2TensorID": self.mappingTable.name2TensorID,
        "tensorID2mapping2D": tensorID2mapping2D,
        "globalIdx2TensorID": self.mappingTable.globalIdx2TensorID,
        "globalIdx2mappingIdx": self.mappingTable.globalIdx2mappingIdx
      },
      "writeCounts": wrtCount
    }
    with open(filepath, "wb") as f:
      torch.save(wl_data, f)


if __name__ == '__main__':
  resManager = ResourceManager()
  mappingTable = MappingTable(resManager)
  wlManager = WearLeveling(resourceManager=resManager, mappingTable=mappingTable)
  old = torch.zeros(size=(4, 4), dtype=cfg.torchInt)
  # new = torch.tensor(
  #   [[1, 1, 0, 1],
  #         [1, 3, 1, 1],
  #         [1, 0, 1, 0],
  #         [0, 1, 3, 0]], dtype=cfg.torchInt)
  new = torch.tensor(
    [[1, 1, 0, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]], dtype=cfg.torchInt)


  nameList = wlManager.initByTensorList([new])
  logicalTensor = mappingTable.getLogicalTensor(tensorName=nameList[0])
  logicalTensor.write(new, flushWriteCounts=True, updateLastCellBit=True)
  wlManager.shifting(shifts=(1, 1))
  wlManager.TIWL()
  logicalTensor.write(old, flushWriteCounts=True, updateLastCellBit=True)
  wlManager.shifting(shifts=(1, 1))
  wlManager.TIWL()
  print("")
