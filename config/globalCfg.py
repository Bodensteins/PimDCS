import os
import json
import math
import torch
import platform
import configparser
import datetime as dt
from enum import Enum

class TensorType(Enum):
  Normal = 0  # two's complement representation: n+1 bits range: -2^n ~ 2^n-1
  # Ref = 1  # abandoned
  PN = 2  # pos neg representation: n bits range: -(2^n-1) ~ 2^n-1
  # SM = 3  # we will support it int the future, sign magnitude representation: n+1 bits range: -(2^n-1) ~ 2^n-1

class RightShiftMode(Enum):
  Abandon = 0
  Round = 1
  RoundToEvenNearest = 2

class WeightUpdateStrategy(Enum):
  StaticRange = 0
  DynamicRange = 1

class OptimMode(Enum):
  FullFix = 0
  FloatWeight = 1
  FullFloat = 2

class PerformanceEvaluateMode(Enum):
  # normal training, transient data used for backward is also stored in phy pim array.
  # refers to Pipelayer(hpca 2017), or Time (dac17) architecture.
  Train = 0
  FastModeTrain = 1  # backend is digital
  Inference = 2
  # transient data used for backward is stored in buffer.
  TrainTransientInBuffer = 3

class GlobalCfg:
  _instance = None
  def __init__(self, config_path = os.path.join(os.path.dirname(__file__), "config.ini")):
    self.config = configparser.ConfigParser()
    self.config.optionxform = str
    self.config.read(config_path)
    self.timeStamp = dt.datetime.now()

    # Quantization
    self.tensorType = TensorType[self.config["Quantization"]["TensorType"]]
    self.rightShiftMode = RightShiftMode[self.config["Quantization"]["RightShiftMode"]]
    self.weightUpdateStrategy = WeightUpdateStrategy[self.config["Quantization"]["WeightUpdateStrategy"]]
    self.dataFlowBitWidth = self.config["Quantization"].getint("DataFlowBitWidth")
    self.outputBitWidth = self.config["Quantization"].getint("OutputBitWidth")
    self.weightBitWidth = self.config["Quantization"].getint("WeightBitWidth")
    self.gradOutputBitWidth = self.config["Quantization"].getint("GradOutputBitWidth")
    self.computeWeightBitWidth = self.config["Quantization"].getint("ComputeWeightBitWidth")
    self.optimMode = OptimMode[self.config["Quantization"]["OptimMode"]]
    self.systemBitWidth = 64 if platform.architecture()[0] == '64bit' else 32
    self.torchInt = torch.int64 if self.systemBitWidth == 64 else torch.int32
    self.torchFloat = torch.float64 if self.systemBitWidth == 64 else torch.float32


    # Simulator.arch
    self.chip_n = self.config["Simulator.arch"].getint("Chip_n")
    self.bank_n = self.config["Simulator.arch"].getint("Bank_n")
    self.tile_n = self.config["Simulator.arch"].getint("Tile_n")
    self.pe_n = self.config["Simulator.arch"].getint("PE_n")
    self.crx_n = self.config["Simulator.arch"].getint("Crx_n")
    self.crxShape = json.loads(self.config["Simulator.arch"]["CrxShape"])
    self.ouShape = json.loads(self.config["Simulator.arch"]["OuShape"])
    self.cellBits = self.config["Simulator.arch"].getint("CellBits")
    self.mmManagerTpye = self.config["Simulator.arch"].getint("MMM_Type")
    assert self.crxShape[0] % self.ouShape[0] == 0 and self.crxShape[1] % self.ouShape[1] == 0
    self.cellNumPerValue = math.ceil(self.weightBitWidth / self.cellBits)
    self.ouMapShape = [self.crxShape[0] // self.ouShape[0], self.crxShape[1] // self.ouShape[1]]
    self.ou_n = self.ouMapShape[0] * self.ouMapShape[1]
    self.archInfo = (self.chip_n, self.bank_n, self.tile_n, self.pe_n, self.crx_n, self.ou_n)

  def updateConfigs(self, configs: dict):
    for k, v in configs.items():
      setattr(self, k, v)

    # update dependency
    assert self.crxShape[0] % self.ouShape[0] == 0 and self.crxShape[1] % self.ouShape[1] == 0
    self.cellNumPerValue = math.ceil(self.weightBitWidth / self.cellBits)
    self.ouMapShape = [self.crxShape[0] // self.ouShape[0], self.crxShape[1] // self.ouShape[1]]
    self.ou_n = self.ouMapShape[0] * self.ouMapShape[1]
    self.archInfo = (self.chip_n, self.bank_n, self.tile_n, self.pe_n, self.crx_n, self.ou_n)

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __str__(self):
    cfgStr = "Configs:\n"
    for k in self.__dict__:
      if k == 'config':
        continue
      cfgStr += "%s: %s\n" % (k, self.__dict__[k])
    return cfgStr

# This is a global variable of configs.
globalCfg = GlobalCfg()