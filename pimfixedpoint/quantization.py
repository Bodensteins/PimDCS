from __future__ import annotations

import math

import torch
import numpy as np
from commonMethod import *


# quantization parameter
class QTensor(torch.Tensor):
    def __init__(self):
        super().__init__()
        # print('QTensor init')
        self.s = None  # the fixed point position
        self.resolution = None
        self.bit_width = None
        self.min = None
        self.max = None
        self.fixed_tensor = None
        self.neg_levels = None  # 2^(bit_width - 1)
        self.pos_levels = None  # 2^(bit_width - 1) - 1

    def init_quantization_info(self, max_abs_value: float, bit_width: int):
        self.bit_width = bit_width
        self.neg_levels = pow_2_n(bit_width - 1)
        self.pos_levels = self.neg_levels - 1
        self.s = get_fixed_point_position(max_abs_value, bit_width)  # s: fixed point position
        self.resolution = math.pow(2, self.s)
        self.min = -self.resolution * self.neg_levels
        self.max = self.resolution * self.pos_levels

    def print_quantization_info(self):
        print(f'tensor type: {self.type()}\n'
              f'fixed point position: {self.s}\n'
              f'quantization resolution: {self.resolution}\n'
              f'bit width: {self.bit_width}\n'
              f'data range: {self.min} ~ {self.max}')

    def quantization(self, tensor: torch.Tensor, max_abs_value, bit_width):
        print_call_abstract_method_info()
        pass

    def de_quantization(self):
        print_call_abstract_method_info()
        pass


class ArrayTensor(QTensor):
    def __init__(self):
        super().__init__()

    # may change later
    def sub(self, other: NormalTensor):
        print_call_abstract_method_info()
        pass

    def write(self, other: NormalTensor):
        print_call_abstract_method_info()
        pass


class NormalTensor(QTensor):
    def __init__(self):
        super().__init__()

    # first layer need to process dataset to initialize
    def quantization(self, tensor: torch.Tensor, max_abs_value: float, bit_width: int):
        self.init_quantization_info(max_abs_value, bit_width)
        self.fixed_tensor = tensor.div(self.resolution).round().to(torch.int32)
        self.data = torch.from_numpy(self.fixed_tensor.numpy().view(dtype=np.float32))

    def de_quantization(self):
        return self.fixed_tensor.mul(self.resolution)

    def mul(self, other: ArrayTensor) -> NormalTensor:
        pass

    def t(self) -> NormalTensor:
        pass

    def t_mul(self, other: ArrayTensor) -> NormalTensor:
        pass


class RefTensor(ArrayTensor):
    def __init__(self):
        super().__init__()

    def quantization(self, tensor: torch.Tensor, max_abs_value, bit_width):
        # max_value = weight.max().item()
        # min_value = weight.min).item()
        self.init_quantization_info(max_abs_value, bit_width)
        self.fixed_tensor = torch.empty([tensor.size(0), tensor.size(1) + 1], dtype=torch.int32)
        self.fixed_tensor[..., -1] = self.neg_levels  # ref col
        self.fixed_tensor[..., 0:-1] = tensor.div(self.resolution).round().to(torch.int32).\
            add(self.neg_levels)
        self.data = torch.from_numpy(self.fixed_tensor.numpy().view(dtype=np.float32))

    def de_quantization(self):
        return self.fixed_tensor[..., 0:-1].sub(self.neg_levels).mul(self.resolution)

    def sub(self, other: NormalTensor):
        pass

    def sub_t(self, other: NormalTensor):
        pass

    def write(self, other: NormalTensor):
        pass

# class PNTensor(ArrayTensor):
#     def __init__(self, row_size, col_size, max_abs_value, bit_width):
#         super().__init__(bit_width, row_size, col_size, 2)
#         self.pos_fixed_tensor = None
#         self.neg_fixed_tensor = None
#         self.s = commonMethod.get_fixed_point_position(max_abs_value, self.bit_width + 1)  # p&n equal add on more bit
#         self.resolution = math.pow(2, self.s)
#         self.min = -self.resolution * commonMethod.pos_levels(self.bit_width + 1)
#         self.max = self.resolution * commonMethod.pos_levels(self.bit_width + 1)
#
#     def quantization_tensor(self, tensor: torch.Tensor):
#         fixed_tensor = tensor.div(self.resolution).round().to(torch.int32)
#         abs_fixed_tensor = fixed_tensor.abs()
#         self.pos_fixed_tensor = abs_fixed_tensor.add(fixed_tensor) >> 1
#         self.neg_fixed_tensor = abs_fixed_tensor.sub(fixed_tensor) >> 1
#
#     def de_quantization(self):
#         return self.pos_fixed_tensor.sub(self.neg_fixed_tensor).mul(self.resolution)
#
#     def sub(self, other: NormalTensor):
#         pass
#
#     def __sub__(self, other: NormalTensor):
#         pass
