import math
import sys

import torch
import numpy as np

from enum import Enum


class TensorType(Enum):
    Normal = 0
    Ref = 1
    PN = 2


class QuantizationPara(object):
    def __init__(self):
        self.s = None  # the fixed point position
        self.bit_width = None
        self.tensor_type = None
        self.resolution = None
        self.neg_levels = None  # 2^(bit_width - 1)
        self.pos_levels = None  # 2^(bit_width - 1) - 1
        self.min_value = None
        self.max_value = None

    def parse_quantization_para_tensor(self, quantization_para_tensor: torch.Tensor):
        self.s = quantization_para_tensor[0].item()
        self.bit_width = quantization_para_tensor[1].item()
        self.tensor_type = TensorType(quantization_para_tensor[2].item())
        self.resolution = pow(2, self.s)
        self.neg_levels = pow_2_n(self.bit_width)
        self.pos_levels = self.neg_levels - 1
        self.min_value = -self.resolution * self.neg_levels
        self.max_value = self.resolution * self.pos_levels

    def set_quantization_para_tensor(self, quantization_para_tensor: torch.Tensor):
        quantization_para_tensor[0] = self.s
        quantization_para_tensor[1] = self.bit_width
        quantization_para_tensor[2] = self.tensor_type.value


def get_fixed_point_position(max_abs: float, bit_width: int) -> int:
    return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))


def pow_2_n(n: int) -> int:
    return 1 << n


def get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1


def fixed_point_to_float(fixed_point_tensor: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(fixed_point_tensor.numpy().view(dtype=np.float32))


def float_to_fixed_point(float_tensor: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(float_tensor.numpy().view(dtype=np.int32))


def parse_float_tensor_list(float_tensor_list: list):
    fixed_point_tensor = float_to_fixed_point(float_tensor_list[0])
    quantization_para_tensor = float_to_fixed_point(float_tensor_list[1])
    return fixed_point_tensor, quantization_para_tensor


def print_call_abstract_method_info():
    print(f'Error: {sys._getframe().f_code.co_name} is an abstract method!')


def print_call_incomplete_method_info():
    print(f'Error: {sys._getframe().f_code.co_name} is an incomplete method!')
