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
    assert self.crxShape[0] % self.ouShape[0] == 0 and self.crxShape[1] % self.ouShape[1] == 0
    self.cellNumPerValue = math.ceil(self.weightBitWidth / self.cellBits)
    self.ouMapShape = [self.crxShape[0] // self.ouShape[0], self.crxShape[1] // self.ouShape[1]]
    self.ou_n = self.ouMapShape[0] * self.ouMapShape[1]
    self.archInfo = (self.chip_n, self.bank_n, self.tile_n, self.pe_n, self.crx_n, self.ou_n)

    # Simulator.dataPlacement
    self.dataSplitting = self.config["Simulator.dataPlacement"].getboolean("DataSplitting")

    # Simulator.staticPerformance
    self.performanceEvaluateMode = PerformanceEvaluateMode[self.config["Simulator.staticPerformance"]["PerformanceEvaluateMode"]]
    self.training = self.config["Simulator.staticPerformance"].getboolean("Training")
    self.epochs = self.config["Simulator.staticPerformance"].getint("Epochs")
    self.dataSize = self.config["Simulator.staticPerformance"].getint("DataSize")
    self.batchSize = self.config["Simulator.staticPerformance"].getint("BatchSize")
    self.netMapTimes = json.loads(self.config["Simulator.staticPerformance"]["NetMapTimes"])
    self.inputDataShape = json.loads(self.config["Simulator.staticPerformance"]["InputDataShape"])
    self.activationUnitNum = self.config["Simulator.staticPerformance"].getint("ActivationUnitNum")
    self.activationUnitLatency = self.config["Simulator.staticPerformance"].getint("ActivationUnitLatency")
    self.arrayMode = self.config["Simulator.staticPerformance"].getint("ArrayMode")

    # Simulator.staticPerformance.latency
    self.computeMode = self.config["Simulator.staticPerformance.latency"]["ComputeMode"]

    # Simulator.staticPerformance.area
    self.singleDACArea = self.config["Simulator.staticPerformance.area"].getfloat("SingleDACArea")
    self.singleADCArea = self.config["Simulator.staticPerformance.area"].getfloat("SingleADCArea")
    self.singleSHArea = self.config["Simulator.staticPerformance.area"].getfloat("SingleSHArea")
    self.singleCellArea = self.config["Simulator.staticPerformance.area"].getfloat("SingleCellArea")
    self.adderArea = self.config["Simulator.staticPerformance.area"].getfloat("AdderArea")
    self.DAC_n = self.config["Simulator.staticPerformance.area"].getint("DAC_n")
    self.ADC_n = self.config["Simulator.staticPerformance.area"].getint("ADC_n")
    self.adder_n = self.config["Simulator.staticPerformance.area"].getint("Adder_n")
    self.OR_n = self.config["Simulator.staticPerformance.area"].getint("OR_n")
    self.IR_n = self.config["Simulator.staticPerformance.area"].getint("IR_n")
    self.SH_n = self.config["Simulator.staticPerformance.area"].getint("SH_n")
    self.SA_n = self.config["Simulator.staticPerformance.area"].getint("SA_n")

    # Wear Leveling
    if self.config.has_section("Simulator.extension.wearLeveling"):
      self.TIWLInterval = self.config["Simulator.extension.wearLeveling"].getint("TIWLInterval")
      self.PEShiftInterval = self.config["Simulator.extension.wearLeveling"].getint("PEShiftInterval")
      self.crxWlInterval = self.config["Simulator.extension.wearLeveling"].getint("CrxWlInterval")
      self.crxWlType = self.config["Simulator.extension.wearLeveling"].get("CrxWlType")
      self.LASRTopK = self.config["Simulator.extension.wearLeveling"].getfloat("LASRTopK")
      self.granularityTIWL = self.config["Simulator.extension.wearLeveling"].get("GranularityOfTIWL").lower()
      self.granularityAlloc = self.config["Simulator.extension.wearLeveling"].get("GranularityOfAllocator").lower()
      self.overlapUpdate = self.config["Simulator.extension.wearLeveling"].getboolean("OverlapUpdate")
      self.fillingType = self.config["Simulator.extension.wearLeveling"].get("FillingType")
      self.fillingParam = self.config["Simulator.extension.wearLeveling"].getfloat("FillingParam")
      self.shiftNum = self.config["Simulator.extension.wearLeveling"].getint("ShiftNum")
      self.storeTranspose = self.config["Simulator.extension.wearLeveling"].getboolean("StoreTranspose")
      self.saveInterval = self.config["Simulator.extension.wearLeveling"].getint("SaveInterval")
      self.saveDir = self.config["Simulator.extension.wearLeveling"].get("SaveDir")
      self.saveLiteWlState = self.config["Simulator.extension.wearLeveling"].getboolean("SaveLiteWlState")
      self.saveJsonMapping = self.config["Simulator.extension.wearLeveling"].getboolean("SaveJsonMapping")

    # Custom
    for key, value in self.config["Custom"].items():
      setattr(self, key, value)

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