from torch.nn import Module
from .. import functional as fpF
from ..commonConst import data_flow_bit_width


class Quan(Module):
    def __init__(self, bit_width=data_flow_bit_width):
        super().__init__()
        self._bit_width = bit_width

    def forward(self, _input):
        return fpF.quan.apply(_input, self._bit_width)


class DeQuan(Module):
    def __init__(self, bit_width: int = data_flow_bit_width, back_bit_width: int = data_flow_bit_width,
                 quantization_mode: str = "dynamic"):
        super().__init__()
        self.quantizationMode = quantization_mode
        self._bit_width = bit_width
        self.backBit = back_bit_width

    def forward(self, qinput, qinput_config):
        return fpF.dequan.apply(qinput, qinput_config, self._bit_width, self.backBit, self.quantizationMode)