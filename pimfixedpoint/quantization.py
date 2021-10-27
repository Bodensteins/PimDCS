from __future__ import annotations

import torch
from torch import Tensor
import numpy as np
import math

from enum import Enum


class TensorType(Enum):
    Normal = 0
    Ref = 1
    PN = 2


def get_fixed_point_position(max_abs: float, bit_width: int) -> int:
    return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))


def pow_2_n(n: int) -> int:
    return 1 << n


def get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1

# input: quantization parameters, should be int Tensor
# return: quantization para. {s, bit_width, tensor_type(Normal or ref or PN)}
def parse_quantization_para(quantization_para: Tensor):
    s = quantization_para[0].item()
    bit_width = quantization_para[1].item()
    tensor_type = TensorType(quantization_para[2].item())

    return s, bit_width, tensor_type


def print_quantization_info(s: int, bit_width: int, tensor_type: TensorType):
    resolution = pow(2, s)
    neg_levels = pow_2_n(bit_width - 1)
    pos_levels = neg_levels - 1
    max_value = pos_levels * resolution
    min_value = -neg_levels * resolution

    print(f'tensor type: {tensor_type}\n'
          f'fixed point position: {s}\n'
          f'quantization resolution: {resolution}\n'
          f'bit width: {bit_width}\n'
          f'neg levels: {neg_levels}\n'
          f'pos levels: {pos_levels}\n'
          f'data range: {min_value} ~ {max_value}')


def int_to_float(int_tensor: Tensor) -> Tensor:
    return torch.from_numpy(int_tensor.numpy().view(dtype=np.float32))


def float_to_int(float_tensor: Tensor) -> Tensor:
    return torch.from_numpy(float_tensor.detach().numpy().view(dtype=np.int32))


system_bit_width = 32
data_flow_bit_width = system_bit_width >> 1
# data flow bit width must be half of system bit width to avoid overflow


def parse_float_tensor_list(float_tensor_list: list):
    if float_tensor_list[0] is not None:
        int_tensor = float_to_int(float_tensor_list[0])
    else:
        int_tensor = None

    quantization_para = float_to_int(float_tensor_list[1])

    return int_tensor, quantization_para


def creat_quantization_para(s: int = None, bit_width: int = None, tensor_type: TensorType = None):
    quantization_para = torch.empty(3, dtype=torch.int32)
    if s is not None:
        quantization_para[0] = s
    if bit_width is not None:
        quantization_para[1] = bit_width
    if tensor_type is not None:
        quantization_para[2] = tensor_type.value

    return int_to_float(quantization_para)


# return int_to_float(quantization_tensor), quantization_para can be modified in this method, don't need to return
def quantization_tensor(quantization_para: Tensor, tensor: Tensor, max_abs_value: float) -> torch.Tensor:
    quantization_para = float_to_int(quantization_para)
    _, bit_width, tensor_type = parse_quantization_para(quantization_para)  # s is unknown

    if tensor_type == TensorType.Normal:
        s = get_fixed_point_position(max_abs_value, bit_width)
        quantization_para[0] = s
        resolution = pow(2, s)
        int_tensor = tensor.div(resolution).round().to(torch.int32)
    elif tensor_type == TensorType.Ref:
        s = get_fixed_point_position(max_abs_value, bit_width)
        quantization_para[0] = s
        resolution = pow(2, s)
        int_tensor = torch.empty([tensor.size()[0], tensor.size()[1] + 1], dtype=torch.int32)
        neg_levels = pow_2_n(bit_width - 1)
        int_tensor[:, -1] = neg_levels
        int_tensor[:, 0:-1] = tensor.div(resolution).round().to(torch.int32).add(neg_levels)
    elif tensor_type == TensorType.PN:
        raise Exception("We don't implement this tensor_type!", tensor_type)
    else:
        raise Exception("Invalid tensor_type!", tensor_type)

    return int_to_float(int_tensor)


def de_quantization(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    if tensor_type == TensorType.Normal:
        resolution = pow(2, s)

        return int_tensor.mul(resolution)
    elif tensor_type == TensorType.Ref:
        resolution = pow(2, s)
        neg_levels = pow_2_n(bit_width - 1)

        return int_tensor[:, 0:-1].sub(neg_levels).mul(resolution)
    elif tensor_type == TensorType.PN:
        raise Exception("We don't implement this tensor_type!", tensor_type)
    else:
        raise Exception("Invalid tensor_type!", tensor_type)


def change_bit_width_(float_tensor_list: list, new_bit_width):
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    if new_bit_width < bit_width:
        quantization_para[0] = s + bit_width - new_bit_width
        int_tensor.__irshift__(bit_width - new_bit_width)

    quantization_para[1] = new_bit_width


# not in-situ method, delete _
def add_additional_col_of_one(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    resolution = pow(2, s)
    max_value = (pow_2_n(bit_width) - 1) * resolution

    int_one = round(1 / resolution)

    if max_value < 1:
        new_bit_width = get_pos_bit_width(int_one)
        change_bit_width_(float_tensor_list, new_bit_width)

    full_one_col = torch.full([int_tensor.size()[0], 1], int_one, dtype=torch.int32)
    int_tensor = torch.cat((int_tensor, full_one_col), 1)

    return int_to_float(int_tensor)


def add_additional_col_of_zero(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    _, _, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    full_zero_col = torch.full([int_tensor.size()[0], 1], 0, dtype=torch.int32)
    int_tensor = torch.cat((int_tensor, full_zero_col), 1)

    return int_to_float(int_tensor)


# don't need?
def remove_additional_col(float_tensor: Tensor):
    float_tensor = float_tensor[:, 0:-1]

    return float_tensor


def write_array(array_tensor_list: list, data_tensor_list: list) -> Tensor:
    array_int_tensor, array_quantization_para = parse_float_tensor_list(array_tensor_list)
    _, array_bit_width, array_tensor_type = parse_quantization_para(array_quantization_para)
    data_int_tensor, data_quantization_para = parse_float_tensor_list(data_tensor_list)
    data_s, data_bit_width, data_tensor_type = parse_quantization_para(data_quantization_para)

    assert (array_tensor_type == TensorType.Ref or array_tensor_type == TensorType.PN)
    assert (data_tensor_type == TensorType.Normal)

    if array_tensor_type == TensorType.Ref:
        neg_levels = pow_2_n(array_bit_width - 1)
        if array_int_tensor is None:
            array_int_tensor = \
                torch.empty([data_int_tensor.size()[0], data_int_tensor.size()[1] + 1], dtype=torch.int32)
            array_int_tensor[:, -1] = neg_levels

        if array_bit_width >= data_bit_width:
            array_quantization_para[0] = data_s
            array_int_tensor[:, 0:-1] = data_int_tensor.add(neg_levels)
        else:
            array_quantization_para[0] = data_s + data_bit_width - array_bit_width
            array_int_tensor[:, 0:-1] = data_int_tensor.__rshift__(data_bit_width - array_bit_width).add(neg_levels)

    else:
        raise Exception("We don't implement this tensor_type!", array_tensor_type)

    return int_to_float(array_int_tensor)


def set_appropriate_bit_width_(float_tensor_list: list):
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    _, _, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    max_int = int_tensor.max().item()
    min_int = int_tensor.min().item()
    bit_width = 2

    if max_int <= 0:
        if min_int != 0:
            bit_width = get_neg_bit_width(min_int)
    elif min_int >= 0:
        bit_width = get_pos_bit_width(max_int)
    else:
        bit_width = max(get_pos_bit_width(max_int), get_neg_bit_width(min_int))

    quantization_para[1] = max(bit_width, 2)  # at least 2 bits


def normal_matmul_array(normal_tensor_list: list, array_tensor_list: list, matmul_result_para: Tensor = None) -> [Tensor, Tensor]:
    normal_int_tensor, normal_quantization_para = parse_float_tensor_list(normal_tensor_list)
    normal_s, normal_bit_width, normal_tensor_type = parse_quantization_para(normal_quantization_para)
    array_int_tensor, array_quantization_para = parse_float_tensor_list(array_tensor_list)
    array_s, array_bit_width, array_tensor_type = parse_quantization_para(array_quantization_para)

    assert (normal_tensor_type == TensorType.Normal)
    assert (array_tensor_type == TensorType.Ref or array_tensor_type == TensorType.PN)

    if array_tensor_type == TensorType.Ref:
        temp_result = normal_int_tensor.matmul(array_int_tensor)
        matmul_result = temp_result[:, 0:-1] - temp_result[:, -1].unsqueeze(0).t()

        if matmul_result_para is None:
            matmul_result_para = creat_quantization_para(s=normal_s + array_s, tensor_type=TensorType.Normal)
        else:
            int_matmul_result_para = float_to_int(matmul_result_para)
            int_matmul_result_para[0] = normal_s + array_s
            assert (int_matmul_result_para[2] == TensorType.Normal.value)

        set_appropriate_bit_width_([matmul_result, matmul_result_para])
    else:
        raise Exception("We don't implement this tensor_type!", array_tensor_type)

    change_bit_width_([matmul_result, matmul_result_para], data_flow_bit_width)
    return int_to_float(matmul_result), int_to_float(matmul_result_para)


def normal_t_matmul_array(normal_tensor_list: list, array_tensor_list: list, matmul_result_para: Tensor = None) -> \
        [Tensor, Tensor]:
    normal_tensor, normal_para = normal_tensor_list

    return normal_matmul_array([normal_tensor.t(), normal_para], array_tensor_list, matmul_result_para)


def quantization_tensor_less(float_tensor_list: list, a: float) -> torch.BoolTensor:
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    if tensor_type == TensorType.Normal:
        int_a = round(a / pow(2, s))

        return int_tensor < int_a
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)


def mul_num(float_tensor_list: list, alpha: float, alpha_bit_width: int = 16) -> [Tensor, Tensor]:
    change_bit_width_(float_tensor_list, 16)
    int_tensor, quantization_para = parse_float_tensor_list(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    alpha_s = get_fixed_point_position(abs(alpha), alpha_bit_width)
    alpha_resolution = pow(2, alpha_s)
    alpha_int = round(alpha / alpha_resolution)

    if tensor_type == TensorType.Normal:
        mul_num_int_tensor = int_tensor.mul(alpha_int)
        mul_num_s = s + alpha_s
        mul_num_para = creat_quantization_para(s=mul_num_s, tensor_type=TensorType.Normal)
        set_appropriate_bit_width_([mul_num_int_tensor, mul_num_para])
    elif tensor_type == TensorType.Ref:
        data_part = int_tensor[..., 0:-1]
        ref_part = int_tensor[..., -1].unsqueeze(0).t()
        mul_num_int_tensor = (data_part - ref_part).mul(alpha_int)
        mul_num_s = s + alpha_s
        mul_num_para = creat_quantization_para(s=mul_num_s, tensor_type=TensorType.Normal)
        set_appropriate_bit_width_([mul_num_int_tensor, mul_num_para])
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)

    change_bit_width_([mul_num_int_tensor, mul_num_para], data_flow_bit_width)
    return int_to_float(mul_num_int_tensor), int_to_float(mul_num_para)


def add_alpha_tensor_(source_tensor_list: list, add_tensor_list: list, alpha: float = 1, alpha_bit_width: int = 16):
    source_int_tensor, source_quantization_para = parse_float_tensor_list(source_tensor_list)
    source_s, source_bit_width, source_tensor_type = parse_quantization_para(source_quantization_para)

    mul_num_tensor, mul_num_para = mul_num(add_tensor_list, alpha, alpha_bit_width)
    change_bit_width_([mul_num_tensor, mul_num_para], 16)
    mul_num_int_tensor, mul_num_quantization_para = parse_float_tensor_list([mul_num_tensor, mul_num_para])
    mul_num_s, mul_num_bit_width, mul_num_tensor_type = parse_quantization_para(mul_num_quantization_para)



    # print_quantization_info(mul_num_s, mul_num_bit_width, mul_num_tensor_type)
    # print(float_to_int(mul_num_int_tensor))
    # print(de_quantization([mul_num_int_tensor, mul_num_para]))

    if source_tensor_type == TensorType.Normal:
        shift = source_s - mul_num_s
        if shift >= 0:
            mul_num_int_tensor.__irshift__(shift)
        else:
            mul_num_int_tensor.__ilshift__(-shift)

        source_int_tensor.add_(mul_num_int_tensor)
        neg_levels = pow_2_n(source_bit_width - 1)
        pos_levels = neg_levels - 1
        source_int_tensor[source_int_tensor < -neg_levels] = -neg_levels
        source_int_tensor[source_int_tensor > pos_levels] = pos_levels
    elif source_tensor_type == TensorType.Ref:
        shift = source_s - mul_num_s
        # data = source_int_tensor[:, 0:-1]
        data = source_int_tensor
        if shift >= 0:
            mul_num_int_tensor.__irshift__(shift)
        else:
            mul_num_int_tensor.__ilshift__(-shift)

        data.add_(mul_num_int_tensor)
        max_int_number = pow_2_n(source_bit_width) - 1
        data[data < 0] = 0
        data[data > max_int_number] = max_int_number
    else:
        raise Exception("We don't implement this source_tensor_type!", source_tensor_type)
