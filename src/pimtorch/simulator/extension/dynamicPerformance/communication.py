# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.simulator.addressParser import ArchLevel
import src.pimtorch.nn.fixedPointArithmetic as fpA

class ArchAddress:
  def __init__(self, levelID, level: ArchLevel):
    self.levelID = levelID
    self.level = ArchLevel(level)

  def __lt__(self, other):
    if self.level == other.level:
      return self.levelID < other.levelID
    else:
      return self.level > other.level

  def __gt__(self, other):
    return other.__lt__(self)

  def __eq__(self, other):
    return self.levelID == other.levelID and self.level == other.level

  def __hash__(self):
    return hash((self.levelID, self.level))

  def __repr__(self):
    return "(LevelID={}, ArchLevel={})".format(self.levelID, self.level.name)


def _vertical_reduction(pid_list):
  comm_cnt = torch.zeros([5], dtype=torch.int64)
  root_level = ArchLevel.offchip
  pid_list = pid_list.tolist()
  for level in range(ArchLevel.crx, ArchLevel.offchip, -1):
    old_length = len(pid_list)
    if old_length > 1:
      comm_cnt[ArchLevel.crx - level] += old_length
      pid_list = list(set(map(lambda pid: pid // cfg.archInfo[level - 1], pid_list)))
    else:
      root_level = level
      break
  root_levelID = pid_list[0]
  for level in range(ArchLevel.crx, root_level, -1):
    root_levelID *= cfg.archInfo[level - 1]
  return comm_cnt, ArchAddress(root_levelID, root_level)

def _reduction(arch_addr_list):
  comm_cnt = torch.zeros([5], dtype=torch.int64)
  final_level = ArchLevel.offchip
  for level in range(ArchLevel.crx, ArchLevel.offchip, -1):
    if len(arch_addr_list) > 1:
      factor = cfg.archInfo[level - 1]
      new_current_level = []
      for addr in arch_addr_list:
        if addr.level == level:
          new_current_level.append(ArchAddress(addr.levelID // factor, addr.level - 1))
          comm_cnt[ArchLevel.crx - level] += 1
        else:
          new_current_level.append(ArchAddress(addr.levelID // factor, addr.level))
      arch_addr_list = list(set(new_current_level))
    else:
      final_level = level
      break
  final_levelID = arch_addr_list[0].levelID
  for level in range(ArchLevel.crx, final_level, -1):
    final_levelID *= cfg.archInfo[level - 1]
  return comm_cnt, ArchAddress(final_levelID, final_level)


def _analyze_communication(addr_tensor, comm_tensor, mapping2D):
  addr_tensor = fpA.to_int(addr_tensor)
  src_addr = addr_tensor[0].item()
  src_level = addr_tensor[1].item()
  root_addrs = []

  # send input data
  arch_addr_list = [ArchAddress(dst_addr.item(), ArchLevel.crx) for dst_addr in mapping2D.flatten()]
  arch_addr_list.append(ArchAddress(src_addr, src_level))
  comm_cnt_tmp, _ = _reduction(arch_addr_list)
  comm_tensor += comm_cnt_tmp

  for stripe in mapping2D:
    comm_cnt_tmp, root_addr = _vertical_reduction(stripe)
    comm_tensor += comm_cnt_tmp
    root_addrs.append(root_addr)
  comm_cnt_tmp, final_addr = _reduction(root_addrs)
  comm_tensor += comm_cnt_tmp
  addr_tensor[0] = final_addr.levelID
  addr_tensor[1] = final_addr.level
  return addr_tensor, comm_tensor

class layer_holder(Function):
  @staticmethod
  def forward(ctx, addr_tensor, comm_tensor, pidMap2D, pidMap2D_t):
    ctx.save_for_backward(comm_tensor, pidMap2D_t)
    addr_tensor, comm_tensor = _analyze_communication(addr_tensor, comm_tensor, pidMap2D)
    return fpA.to_float(addr_tensor)

  @staticmethod
  def backward(ctx, addr_tensor):
    comm_tensor, pidMap2D_t = ctx.saved_tensors
    addr_tensor, comm_tensor = _analyze_communication(addr_tensor, comm_tensor, pidMap2D_t)
    return fpA.to_float(addr_tensor), None, None, None

class LayerHolder(Module):
  def __init__(self, unique_key, mapping2D: torch.Tensor, mapping2D_t: torch.Tensor):
    super(LayerHolder, self).__init__()
    self.unique_key = unique_key
    self.mapping2D = mapping2D
    self.mapping2D_t = mapping2D_t
    self.comm_tensor = torch.zeros([5], dtype=torch.int64)

  def forward(self, addr_tensor: Tensor):
    addr_tensor = layer_holder.apply(addr_tensor, self.comm_tensor, self.mapping2D, self.mapping2D_t)
    return addr_tensor


class SequentialCommModule(nn.Module):
  def __init__(self, model, mappingTable, store_transpose=True):
    super().__init__()
    self.store_transpose = store_transpose
    self.mappingTable = mappingTable
    comm_layers = []
    for name, layer in model.named_modules():
      if hasattr(layer, "fp_weight"):
        if self.store_transpose:
          comm_layers.append(
            LayerHolder(
              unique_key=name,
              mapping2D=torch.tensor(self.mappingTable.getLogicalTensor(tensorName=name).mapping2D, dtype=torch.int64),
              mapping2D_t=torch.tensor(self.mappingTable.getLogicalTensor(tensorName=name+".t").mapping2D, dtype=torch.int64))
          )
        else:
          comm_layers.append(
            LayerHolder(
              unique_key=name,
              mapping2D=torch.tensor(self.mappingTable.getLogicalTensor(tensorName=name).mapping2D, dtype=torch.int64),
              mapping2D_t=torch.tensor(self.mappingTable.getLogicalTensor(tensorName=name).mapping2D, dtype=torch.int64))
          )
    self.sequential = nn.Sequential(*comm_layers)

  def forward(self, addr_tensor):
    addr_tensor.requires_grad = True
    addr_tensor = self.sequential(addr_tensor)
    return addr_tensor

  def reset_mapping(self, mappingState=None):
    for layer in self.modules():
      if hasattr(layer, "mapping2D"):
        if mappingState is None:
          layer.mapping2D = torch.tensor(self.mappingTable.getLogicalTensor(tensorName=layer.unique_key).mapping2D, dtype=torch.int64)
          if self.store_transpose:
            layer.mapping2D_t = torch.tensor(self.mappingTable.getLogicalTensor(tensorName=layer.unique_key+".t").mapping2D, dtype=torch.int64)
          else:
            layer.mapping2D_t = torch.tensor(self.mappingTable.getLogicalTensor(tensorName=layer.unique_key).mapping2D, dtype=torch.int64)
        else:
          layer.mapping2D = torch.tensor(mappingState[layer.unique_key], dtype=torch.int64)
          if self.store_transpose:
            layer.mapping2D_t = torch.tensor(mappingState[layer.unique_key+".t"], dtype=torch.int64)
          else:
            layer.mapping2D_t = torch.tensor(mappingState[layer.unique_key], dtype=torch.int64)
        layer.comm_tensor = torch.zeros([5], dtype=torch.int64)


  def get_comm_tensor(self):
    comm_tensor = torch.zeros([5], dtype=torch.int64)
    for name, layer in self.named_modules():
      if hasattr(layer, "comm_tensor"):
        comm_tensor += layer.comm_tensor
    return comm_tensor


class CommAnalyzer:
  def __init__(self, dnnModule: torch.nn.Module, mappingTable):
    self.storeTranspose = cfg.storeTranspose
    self.comm_model = SequentialCommModule(model=dnnModule, mappingTable=mappingTable, store_transpose=self.storeTranspose)
    self.comm_data = []

  def analysis(self):
    self.comm_model.reset_mapping(mappingState=None)
    addr_tensor = fpA.to_float(torch.tensor([0, ArchLevel.offchip], dtype=torch.int64))
    f_out_addr_tensor = self.comm_model(addr_tensor)
    if self.storeTranspose:
      f_out_addr_tensor.backward(f_out_addr_tensor)
    self.comm_data.append(self.comm_model.get_comm_tensor())

  def getCommData(self):
    return self.comm_data

if __name__ == '__main__':
  cfg.updateConfigs({
    "crx_n": 4,
    "pe_n": 2,
    "tile_n": 2,
    "bank_n": 2,
    "chip_n": 2
  })
  addr_list = [ArchAddress(0 + 32, ArchLevel.bank),
               ArchAddress(5 + 32, ArchLevel.crx),
               ArchAddress(6 + 32, ArchLevel.crx),
               ArchAddress(2 + 32, ArchLevel.pe),
               ArchAddress(4 + 32, ArchLevel.tile),]
  comm_cnt_tmp, final_addr = _reduction(addr_list)
  print(addr_list)
  print(comm_cnt_tmp)
  print(final_addr)


