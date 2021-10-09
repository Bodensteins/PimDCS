from __future__ import annotations

import torch
import numpy as np
from commonMethod import get_fixed_point_position, pow_2_n, get_pos_bit_width, get_neg_bit_width, \
    print_call_abstract_method_info
import math


# quantization parameter
class QTensor(torch.Tensor):
    def __init__(self):
        super().__init__()
        # print('QTensor init')
        self.s = None  # the fixed point position
        self.resolution = None
        self.bit_width = None
        self.neg_levels = None  # 2^(bit_width - 1)
        self.pos_levels = None  # 2^(bit_width - 1) - 1
        self.min_value = None
        self.max_value = None
        self.fixed_tensor = None

    def init_quantization_info(self, max_abs_value: float):
        self.s = get_fixed_point_position(max_abs_value, self.bit_width)  # s: fixed point position
        self.resolution = math.pow(2, self.s)
        self.min_value = -self.resolution * self.neg_levels
        self.max_value = self.resolution * self.pos_levels

    def print_quantization_info(self):
        print(f'fixed point position: {self.s}\n'
              f'quantization resolution: {self.resolution}\n'
              f'bit width: {self.bit_width}\n'
              f'neg levels: {self.neg_levels}\n'
              f'pos levels: {self.pos_levels}\n'
              f'data range: {self.min_value} ~ {self.max_value}')

    # def quantization(self, tensor: torch.Tensor, max_abs_value: float, bit_width=None):
    #     print_call_abstract_method_info()
    #     pass
    #
    def de_quantization(self):
        print_call_abstract_method_info()
        pass

    def bind_fixed_tensor(self):
        self.data = torch.from_numpy(self.fixed_tensor.numpy().view(dtype=np.float32))


class ArrayTensor(QTensor):
    def __init__(self, bit_width: int):
        super().__init__()
        self.bit_width = bit_width
        self.neg_levels = pow_2_n(bit_width - 1)
        self.pos_levels = self.neg_levels - 1
        self.array_initialized = False

    # may change later
    def sub_normal(self, other: NormalTensor):
        print_call_abstract_method_info()
        pass

    def write_normal(self, other: NormalTensor):
        print_call_abstract_method_info()
        pass


class NormalTensor(QTensor):
    def __init__(self, requires_grad = True):
        super().__init__()
        self.requires_grad_(requires_grad)

    # first layer need to process dataset to initialize
    def quantization(self, tensor: torch.Tensor, max_abs_value: float, bit_width: int):
        self.bit_width = bit_width
        self.neg_levels = pow_2_n(bit_width - 1)
        self.pos_levels = self.neg_levels - 1
        self.init_quantization_info(max_abs_value)
        self.fixed_tensor = tensor.div(self.resolution).round().to(torch.int32)
        self.bind_fixed_tensor()
        self.requires_grad_(tensor.requires_grad)

    def de_quantization(self):
        return self.fixed_tensor.mul(self.resolution)

    def change_bit_width(self, new_bit_width):
        if new_bit_width < self.bit_width:
            self.s = self.s + self.bit_width - new_bit_width
            self.resolution = pow(2, self.s)
            self.fixed_tensor.__irshift__(self.bit_width - new_bit_width)

        self.bit_width = new_bit_width
        self.neg_levels = pow_2_n(self.bit_width - 1)
        self.pos_levels = self.neg_levels - 1
        self.min_value = -self.resolution * self.neg_levels
        self.max_value = self.resolution * self.pos_levels

    def add_additional_one(self):
        fixed_one = round(1 / self.resolution)

        if self.max_value < 1:
            #print("add_additional_one err!")
            new_bit_width = get_pos_bit_width(fixed_one)
            self.change_bit_width(new_bit_width)

        full_one_col = torch.full([self.fixed_tensor.size()[0], 1], fixed_one, dtype=torch.int32)
        self.fixed_tensor = torch.cat((self.fixed_tensor, full_one_col), 1)
        self.bind_fixed_tensor()

    def remove_additional_one(self):
        self.fixed_tensor = self.fixed_tensor[..., 0:-1]
        self.bind_fixed_tensor()

    def normal_t(self) -> NormalTensor:
        other = NormalTensor()
        other.bit_width = self.bit_width
        other.neg_levels = self.neg_levels
        other.pos_levels = self.pos_levels
        other.s = self.s
        other.resolution = self.resolution
        other.min_value = self.min_value
        other.max_value = self.max_value
        other.fixed_tensor = self.fixed_tensor.t().clone()
        other.bind_fixed_tensor()
        return other

    def set_appropriate_bit_width(self):
        max_int = self.fixed_tensor.max().item()
        min_int = self.fixed_tensor.min().item()
        if max_int <= 0:
            self.bit_width = max(get_neg_bit_width(min_int), 1)
        elif min_int >= 0:
            self.bit_width = get_pos_bit_width(max_int)
        else:
            self.bit_width = max(get_pos_bit_width(max_int), get_neg_bit_width(min_int))
        self.neg_levels = pow_2_n(self.bit_width - 1)
        self.pos_levels = self.neg_levels - 1
        self.min_value = -self.resolution * self.neg_levels
        self.max_value = self.resolution * self.pos_levels

    def matmul_array(self, other: ArrayTensor) -> NormalTensor:
        if isinstance(other, RefTensor):
            matmul_result = NormalTensor()
            temp_result = self.fixed_tensor.matmul(other.fixed_tensor)
            # matmul_result.fixed_tensor \
            #     = temp_result[..., 0:-1] - torch.reshape(temp_result[..., -1], [temp_result[..., -1].size()[0], 1])
            matmul_result.fixed_tensor = temp_result[..., 0:-1] - temp_result[..., -1].unsqueeze(0).t()
            matmul_result.bind_fixed_tensor()
            matmul_result.s = self.s + other.s
            matmul_result.resolution = math.pow(2, matmul_result.s)
            matmul_result.set_appropriate_bit_width()
            return matmul_result
        else:
            print("we don't support it!")
            pass

    def t_matmul_array(self, other: ArrayTensor) -> NormalTensor:
        if isinstance(other, RefTensor):
            t_matmul_result = NormalTensor()
            temp_result = self.fixed_tensor.t().matmul(other.fixed_tensor)
            # t_matmul_result.fixed_tensor \
            #     = temp_result[..., 0:-1] - torch.reshape(temp_result[..., -1], [temp_result[..., -1].size()[0], 1])
            t_matmul_result.fixed_tensor = temp_result[..., 0:-1] - temp_result[..., -1].unsqueeze(0).t()
            t_matmul_result.bind_fixed_tensor()
            t_matmul_result.s = self.s + other.s
            t_matmul_result.resolution = math.pow(2, t_matmul_result.s)
            t_matmul_result.set_appropriate_bit_width()
            return t_matmul_result
        else:
            print("we don't support it!")
            pass

    def mul_num(self, alpha: float, alpha_bit_width: int) -> NormalTensor:
        mul_num_result = NormalTensor()
        alpha_s = get_fixed_point_position(abs(alpha), alpha_bit_width)
        alpha_resolution = pow(2, alpha_s)
        alpha_fixed_point = round(alpha / alpha_resolution)
        mul_num_result.fixed_tensor = self.fixed_tensor.mul(alpha_fixed_point)
        mul_num_result.bind_fixed_tensor()
        mul_num_result.s = self.s + alpha_s
        mul_num_result.resolution = pow(2, mul_num_result.s)
        mul_num_result.set_appropriate_bit_width()
        return mul_num_result

    def add_array_(self, other: ArrayTensor, alpha: float, alpha_bit_width: int):
        mul_num_result = other.mul_num(alpha, alpha_bit_width)

        shift = self.s - mul_num_result.s
        data = self.fixed_tensor
        if shift >= 0:
            mul_num_result.fixed_tensor.__irshift__(shift)
        else:
            mul_num_result.fixed_tensor.__ilshift__(-shift)

        data += mul_num_result.fixed_tensor
        data[data < -self.neg_levels] = self.neg_levels
        data[data > self.pos_levels] = self.pos_levels


class RefTensor(ArrayTensor):
    def __init__(self, bit_width: int):
        super().__init__(bit_width)
        self.max_int_number = pow_2_n(bit_width) - 1

    def quantization(self, tensor: torch.Tensor, max_abs_value):
        # max_value = weight.max().item()
        # min_value = weight.min).item()
        self.array_initialized = True
        self.init_quantization_info(max_abs_value)
        self.fixed_tensor = torch.empty([tensor.size()[0], tensor.size()[1] + 1], dtype=torch.int32)
        self.fixed_tensor[..., -1] = self.neg_levels  # ref col
        self.fixed_tensor[..., 0:-1] = tensor.div(self.resolution).round().to(torch.int32).\
            add(self.neg_levels)
        self.bind_fixed_tensor()

    def de_quantization(self):
        return self.fixed_tensor[..., 0:-1].sub(self.neg_levels).mul(self.resolution)

    def sub_normal(self, other: NormalTensor):
        shift = self.s - other.s
        data = self.fixed_tensor[..., 0:-1]
        if shift >= 0:
            other_data = other.fixed_tensor.__rshift__(shift)
        else:
            other_data = other.fixed_tensor.__lshift__(-shift)
        
        data -= other_data
        data[data < 0] = 0
        data[data > self.max_int_number] = self.max_int_number

    def sub_normal_t(self, other: NormalTensor):
        shift = self.s - other.s
        data = self.fixed_tensor[..., 0:-1]
        if shift >= 0:
            other_data = other.fixed_tensor.__rshift__(shift).t()
        else:
            other_data = other.fixed_tensor.__lshift__(-shift).t()
        
        data -= other_data
        data[data < 0] = 0
        data[data > self.max_int_number] = self.max_int_number

    def write_normal(self, other: NormalTensor):
        if not self.array_initialized:
            self.array_initialized = True
            self.fixed_tensor = torch.empty([other.size()[0], other.size()[1] + 1], dtype=torch.int32)
            self.fixed_tensor[..., -1] = self.neg_levels
            self.bind_fixed_tensor()

        if self.bit_width >= other.bit_width:
            self.s = other.s
            self.resolution = other.resolution
            self.fixed_tensor[..., 0:-1] = other.fixed_tensor.add(self.neg_levels)
        else:
            self.s = other.s + other.bit_width - self.bit_width
            self.resolution = pow(2, self.s)
            self.fixed_tensor[..., 0:-1] = other.fixed_tensor.__rshift__(other.bit_width - self.bit_width)\
                .add(self.neg_levels)

        self.min_value = -self.resolution * self.neg_levels
        self.max_value = self.resolution * self.pos_levels

    def mul_num(self, alpha: float, alpha_bit_width: int) -> NormalTensor:
        mul_num_result = NormalTensor()
        alpha_s = get_fixed_point_position(abs(alpha), alpha_bit_width)
        alpha_resolution = pow(2, alpha_s)
        alpha_fixed_point = round(alpha / alpha_resolution)
        # print(f'fixed point: {alpha_fixed_point} alpha_s: {alpha_s} alpha_resolution: {alpha_resolution}')
        data_part = self.fixed_tensor[..., 0:-1]
        ref_part = self.fixed_tensor[..., -1].unsqueeze(0).t()
        mul_num_result.fixed_tensor = (data_part - ref_part).mul(alpha_fixed_point)
        mul_num_result.bind_fixed_tensor()
        mul_num_result.s = self.s + alpha_s
        mul_num_result.resolution = pow(2, mul_num_result.s)
        mul_num_result.set_appropriate_bit_width()
        return mul_num_result

    def add_normal_(self, other: NormalTensor, alpha: float, alpha_bit_width: int):
        mul_num_result = other.mul_num(alpha, alpha_bit_width)

        shift = self.s - mul_num_result.s
        data = self.fixed_tensor[..., 0:-1]
        if shift >= 0:
            mul_num_result.fixed_tensor.__irshift__(shift)
        else:

            mul_num_result.fixed_tensor.__ilshift__(-shift)

        data += mul_num_result.fixed_tensor
        data[data < 0] = 0
        data[data > self.max_int_number] = self.max_int_number

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
