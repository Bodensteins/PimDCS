from __future__ import annotations

import math
import abc
from abc import ABC

import torch
import commonMethod


# quantization parameter
class QTensor(ABC):
    def __init__(self, bit_width: int, row_size: int, col_size: int, tensor_num: int):
        self.s = None  # the fixed point position
        self.resolution = None
        self.bit_width = bit_width
        self.min = None
        self.max = None
        self.row_size = row_size
        self.col_size = col_size
        self.tensor_num = tensor_num  # the fixed tensor number

    def print_quantization_info(self):
        print(f'fixed point position: {self.s}\n'
              f'quantization resolution: {self.resolution}\n'
              f'bit width: {self.bit_width}\n'
              f'data range: {self.min} ~ {self.max}')

    @abc.abstractmethod
    def quantization_tensor(self, tensor: torch.Tensor):
        pass

    @abc.abstractmethod
    def de_quantization(self):
        pass


class ArrayTensor(QTensor, ABC):
    def __init__(self, bit_width: int, row_size: int, col_size: int, tensor_num: int):
        super().__init__(bit_width, row_size, col_size, tensor_num)

    @abc.abstractmethod
    def sub(self, other: NormalTensor):
        pass

    @abc.abstractmethod
    def __sub__(self, other: NormalTensor):
        pass


class NormalTensor(QTensor):
    def __init__(self, row_size: int, col_size: int, max_abs_value: float, bit_width: int):
        super().__init__(bit_width, row_size, col_size, 1)
        self.fixed_tensor = None
        self.s = commonMethod.get_fixed_point_position(max_abs_value, self.bit_width)  # s: fixed point position
        self.resolution = math.pow(2, self.s)
        self.min = -self.resolution * commonMethod.neg_levels(self.bit_width)
        self.max = self.resolution * commonMethod.pos_levels(self.bit_width)

    # first layer need to process dataset to initialize
    def quantization_tensor(self, tensor: torch.Tensor):
        self.fixed_tensor = tensor.div(self.resolution).round().to(torch.int32)

    def de_quantization(self):
        return self.fixed_tensor.mul(self.resolution)

    def mul(self, other: ArrayTensor) -> NormalTensor:
        pass

    def __mul__(self, other: ArrayTensor) -> NormalTensor:
        pass


class RefTensor(ArrayTensor):
    def __init__(self, row_size, col_size, max_abs_value, bit_width):
        super().__init__(bit_width, row_size, col_size + 1, 1)
        self.fixed_tensor = torch.zeros([row_size, col_size + 1], dtype=torch.int32)
        self.fixed_tensor[..., -1] = commonMethod.neg_levels(self.bit_width)  # ref col
        self.s = commonMethod.get_fixed_point_position(max_abs_value, self.bit_width)  # s: fixed point position
        self.resolution = math.pow(2, self.s)
        self.min = -self.resolution * commonMethod.neg_levels(self.bit_width)
        self.max = self.resolution * commonMethod.pos_levels(self.bit_width)

    def quantization_tensor(self, tensor: torch.Tensor):
        # max_value = weight.max().item()
        # min_value = weight.min).item()
        # self.fixed_tensor[..., 0:-1] = tensor.div(self.resolution).round().to(torch.int32)\
        #     .add(self.fixed_tensor[..., -1])
        self.fixed_tensor[..., 0:-1] = tensor.div(self.resolution).round().to(torch.int32).\
            add(commonMethod.neg_levels(self.bit_width))

    def de_quantization(self):
        return self.fixed_tensor[..., 0:-1].sub(commonMethod.neg_levels(self.bit_width)).mul(self.resolution)

    def sub(self, other: NormalTensor):
        pass

    def __sub__(self, other: NormalTensor):
        pass


class PNTensor(ArrayTensor):
    def __init__(self, row_size, col_size, max_abs_value, bit_width):
        super().__init__(bit_width, row_size, col_size, 2)
        self.pos_fixed_tensor = None
        self.neg_fixed_tensor = None
        self.s = commonMethod.get_fixed_point_position(max_abs_value, self.bit_width + 1)  # p&n equal add on more bit
        self.resolution = math.pow(2, self.s)
        self.min = -self.resolution * commonMethod.pos_levels(self.bit_width + 1)
        self.max = self.resolution * commonMethod.pos_levels(self.bit_width + 1)

    def quantization_tensor(self, tensor: torch.Tensor):
        fixed_tensor = tensor.div(self.resolution).round().to(torch.int32)
        abs_fixed_tensor = fixed_tensor.abs()
        self.pos_fixed_tensor = abs_fixed_tensor.add(fixed_tensor) >> 1
        self.neg_fixed_tensor = abs_fixed_tensor.sub(fixed_tensor) >> 1

    def de_quantization(self):
        return self.pos_fixed_tensor.sub(self.neg_fixed_tensor).mul(self.resolution)

    def sub(self, other: NormalTensor):
        pass

    def __sub__(self, other: NormalTensor):
        pass
