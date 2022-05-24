from enum import Enum
import torch


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


system_bit_width = 64
data_flow_bit_width = 32
half_data_flow_bit_width = 16
# data flow bit width must be half of system bit width to avoid overflow
if system_bit_width <= 32:
    torch_int = torch.int32
    torch_float = torch.float32
else:
    torch_int = torch.int64
    torch_float = torch.float64