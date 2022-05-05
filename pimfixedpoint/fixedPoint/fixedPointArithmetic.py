from __future__ import annotations

import torch
from torch import Tensor
import math

from enum import Enum


class TensorType(Enum):
    Normal = 0
    Ref = 1
    PN = 2


class RightShiftMode(Enum):
    Abandon = 0
    Round = 1


class WeightUpdateStrategy(Enum):
    StaticRange = 0
    DynamicRange = 1


abandon_bit_with = None

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


def to_float(tensor):
    return tensor.view(dtype=torch_float)


def to_int(tensor):
    return tensor.detach().view(dtype=torch_int)


# int_tensor matmul int_tensor is not supported in pytorch now(2021/11/6)
def matmul_int_cuda(int_tensor1, int_tensor2):
    assert (int_tensor1.is_cuda and int_tensor2.is_cuda)

    float_tensor1 = int_tensor1.to(dtype=torch_float)
    float_tensor2 = int_tensor2.to(dtype=torch_float)
    mul_result = float_tensor1.matmul(float_tensor2)

    return mul_result.to(dtype=torch_int)


def get_fixed_point_position(max_abs: float, bit_width: int) -> int:
    return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))


def pow_2_n(n: int) -> int:
    return 1 << n


def get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1


def round_rshift_(int_tensor, shift: int):
    assert (shift > 0)
    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    int_tensor.add_(round_bit).__irshift__(shift)


def round_rshift(int_tensor, shift: int):
    if shift <= 0 or shift > system_bit_width - 1:
        raise Exception("Inappropriate shift value: ", shift)
    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    return int_tensor.add(round_bit).__rshift__(shift)


def parse_quantization_para(quantization_para):
    """"
    input: quantization parameters, should be int Tensor
    return: quantization para. {s, bit_width, tensor_type(Normal or ref or PN)}
    """
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


def parse_tensor_list_to_int(tensor_list: list):
    int_tensor = to_int(tensor_list[0])
    quantization_para = to_int(tensor_list[1])

    return int_tensor, quantization_para


def creat_quantization_para(s: int = None, bit_width: int = None, tensor_type: TensorType = None,
                            device: torch.device = torch.device("cpu")):
    quantization_para = torch.empty(3, dtype=torch_int, device=device)
    if s is not None:
        quantization_para[0] = s
    if bit_width is not None:
        quantization_para[1] = bit_width
    if tensor_type is not None:
        quantization_para[2] = tensor_type.value

    return to_float(quantization_para)


def quantization_tensor(quantization_para: Tensor, tensor: Tensor, user_set_max_left: float = None,
                        user_set_max_right: float = None) -> Tensor:
    """
    float-int type: int type, but shown as float
    int-float type: float type, but shown as int
    :param quantization_para: parameters for quantization, which is a float-int type tensor.
    This tensor has 3 values. [0]: s [1]: bit_width [2]: tensor_type
    :param tensor: Float type tensor, that is the real number for you to quantize
    :param user_set_max_left: float, the max abs value used for quantization
    :param user_set_max_right: float, the max abs value used for quantization
    :return: The float-int type tensor. the quantization of input tensor.
    """
    quantization_para = to_int(quantization_para)
    _, bit_width, tensor_type = parse_quantization_para(quantization_para)  # s is unknown

    max_abs_value = tensor.abs().max().item()
    ori_max = max_abs_value
    if user_set_max_right is not None:
        max_abs_value = min(max_abs_value, user_set_max_right)
    if user_set_max_left is not None:
        max_abs_value = max(max_abs_value, user_set_max_left)

    # special deal, if max_abs_value is not the maximum of the tensor.
    if ori_max > max_abs_value:
        tensor = tensor.detach().clone()
        tensor[tensor > max_abs_value] = max_abs_value
        tensor[tensor < -max_abs_value] = -max_abs_value

    s = get_fixed_point_position(max_abs_value, bit_width)
    quantization_para[0] = s
    resolution = pow(2, s)

    if tensor_type == TensorType.Normal:
        int_tensor = tensor.div(resolution).round().to(torch_int)
    elif tensor_type == TensorType.Ref:
        int_tensor = torch.empty([tensor.size()[0], tensor.size()[1] + 1], dtype=torch_int)
        neg_levels = pow_2_n(bit_width - 1)
        int_tensor[:, -1] = neg_levels
        int_tensor[:, 0:-1] = tensor.div(resolution).round().to(torch_int).add(neg_levels)
    elif tensor_type == TensorType.PN:
        int_tensor = torch.empty([2, tensor.size()[0], tensor.size()[1]], dtype=torch_int)
        # P:int_tensor[0] N:int_tensor[1]
        tensor_abs = tensor.abs()
        int_tensor[0] = tensor.add(tensor_abs).div(2)
        int_tensor[1] = tensor.sub(tensor_abs).div(2)
    else:
        raise Exception("Invalid tensor_type!", tensor_type)

    return to_float(int_tensor)


def de_quantization(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    resolution = pow(2, s)
    if tensor_type == TensorType.Normal:
        return int_tensor.to(torch_float).mul(resolution)
    elif tensor_type == TensorType.Ref:
        neg_levels = int_tensor[0, -1].item()

        return int_tensor[:, 0:-1].sub(neg_levels).to(torch_float).mul(resolution)
    elif tensor_type == TensorType.PN:
        return int_tensor[0].sub(int_tensor[1]).mul(resolution)
    else:
        raise Exception("Invalid tensor_type!", tensor_type)


def get_element_wise_effective_bit_width(int_tensor, tensor_type):
    assert (int_tensor.dtype == torch_int)
    if tensor_type is TensorType.Ref:
        bias = int_tensor[0, -1].item()
        int_tensor = int_tensor.sub(bias)
        int_tensor = int_tensor[:, 0:-1]

    bit_width_dict = {}
    for element in int_tensor.flatten():
        element = element.item()
        while element != 0 and element % 2 == 0:
            element >>= 1

        if element < 0:
            bit_width = math.ceil(math.log2(-element))
        else:
            bit_width = math.ceil(math.log2(element + 1))
        if bit_width == 16:
            print(f"element: {element}")
        bit_width += 1
        if bit_width in bit_width_dict:
            bit_width_dict[bit_width] = bit_width_dict[bit_width] + 1
        else:
            bit_width_dict[bit_width] = 1

    return bit_width_dict


def get_effective_bit_width(float_tensor_list: list):
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    _, tensor_bit_width, tensor_type = parse_quantization_para(quantization_para)

    if tensor_type == TensorType.Ref:
        neg_levels = int_tensor[0, -1].item()
        int_tensor = int_tensor[:, 0:-1].sub(neg_levels)

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

    return bit_width


def set_bit_width_(float_tensor_list: list, new_bit_width: int, mode: RightShiftMode = RightShiftMode.Abandon):
    effective_bit_width = get_effective_bit_width(float_tensor_list)

    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    neg_levels = 0

    if tensor_type == TensorType.Ref:
        neg_levels = int_tensor[0, -1].item()
        int_tensor = int_tensor[:, 0:-1]
        int_tensor.sub_(neg_levels)

    if new_bit_width < effective_bit_width:
        quantization_para[0] = s + effective_bit_width - new_bit_width

        if mode == RightShiftMode.Abandon:
            int_tensor.__irshift__(effective_bit_width - new_bit_width)
        elif mode == RightShiftMode.Round:
            round_rshift_(int_tensor, effective_bit_width - new_bit_width)
        else:
            raise Exception("We don't support this right shift mode!", mode)

    if tensor_type == TensorType.Ref:
        int_tensor.add_(neg_levels)

    quantization_para[1] = new_bit_width


def add_additional_col_of_one(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    resolution = pow(2, s)
    max_value = (pow_2_n(bit_width) - 1) * resolution

    int_one = round(1 / resolution)

    if max_value < 1:
        new_bit_width = get_pos_bit_width(int_one)
        set_bit_width_(float_tensor_list, new_bit_width)

    full_one_col = torch.full([int_tensor.size()[0], 1], int_one, dtype=torch_int, device=int_tensor.device)
    int_tensor = torch.cat((int_tensor, full_one_col), 1)

    return to_float(int_tensor)


def add_additional_col_of_zero(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    _, _, tensor_type = parse_quantization_para(quantization_para)

    assert (tensor_type == TensorType.Normal)

    full_zero_col = torch.full([int_tensor.size()[0], 1], 0, dtype=torch_int, device=int_tensor.device)
    int_tensor = torch.cat((int_tensor, full_zero_col), 1)

    return to_float(int_tensor)


def remove_additional_col(tensor):
    return tensor[:, 0:-1]


def write_array_(array_tensor_list: list, data_tensor_list: list, mode: RightShiftMode = RightShiftMode.Abandon):
    data_bit_width = get_effective_bit_width(data_tensor_list)
    array_int_tensor, array_quantization_para = parse_tensor_list_to_int(array_tensor_list)
    _, array_bit_width, array_tensor_type = parse_quantization_para(array_quantization_para)
    data_int_tensor, data_quantization_para = parse_tensor_list_to_int(data_tensor_list)
    data_s, _, data_tensor_type = parse_quantization_para(data_quantization_para)

    assert (array_tensor_type == TensorType.Ref or array_tensor_type == TensorType.PN)
    assert (data_tensor_type == TensorType.Normal)

    if array_tensor_type == TensorType.Ref:
        neg_levels = array_int_tensor[0, -1].item()

        if array_bit_width >= data_bit_width:
            array_quantization_para[0] = data_s
        else:
            array_quantization_para[0] = data_s + data_bit_width - array_bit_width

            if mode == RightShiftMode.Abandon:
                data_int_tensor = data_int_tensor.__rshift__(data_bit_width - array_bit_width)
            elif mode == RightShiftMode.Round:
                data_int_tensor = round_rshift(data_int_tensor, data_bit_width - array_bit_width)
            else:
                raise Exception("We don't support this right shift mode!", mode)

        if array_int_tensor.shape[0] == data_int_tensor.shape[0]:
            array_int_tensor[:, 0:-1].zero_().add_(data_int_tensor.add(neg_levels))
        else:
            array_int_tensor[0:data_int_tensor.shape[0], 0:-1].zero_().add_(data_int_tensor.add(neg_levels))
    else:
        raise Exception("We don't implement this tensor_type!", array_tensor_type)

    return to_float(array_int_tensor)


def quantization_matmul(normal_tensor_list: list, array_tensor_list: list, matmul_result_para: Tensor = None,
                        mask_bit_width: int = abandon_bit_with) -> [Tensor, Tensor]:
    normal_int_tensor, normal_quantization_para = parse_tensor_list_to_int(normal_tensor_list)
    normal_s, normal_bit_width, normal_tensor_type = parse_quantization_para(normal_quantization_para)
    array_int_tensor, array_quantization_para = parse_tensor_list_to_int(array_tensor_list)
    array_s, array_bit_width, array_tensor_type = parse_quantization_para(array_quantization_para)

    assert (normal_tensor_type == TensorType.Normal)

    if normal_int_tensor.shape[1] != array_int_tensor.shape[0]:
        zero_cols = torch.zeros([normal_int_tensor.shape[0], array_int_tensor.shape[0] - normal_int_tensor.shape[1]],
                                dtype=normal_int_tensor.dtype, device=normal_int_tensor.device)
        normal_int_tensor = torch.cat((normal_int_tensor, zero_cols), 1)

    if mask_bit_width is None:
        if normal_int_tensor.is_cuda:
            matmul_result = matmul_int_cuda(normal_int_tensor, array_int_tensor)
        else:
            matmul_result = torch.matmul(normal_int_tensor, array_int_tensor)
    else:
        if mask_bit_width < 0:
            raise Exception("mask bit with < 0!", mask_bit_width)
        elif mask_bit_width > array_bit_width - 2:
            raise Exception("mask bit with too big!", mask_bit_width)

        mask = (-1) & ((-1) << mask_bit_width)  # 11111000...00(0 numbers: mask_bit_width)
        if normal_int_tensor.is_cuda:
            matmul_result = matmul_int_cuda(normal_int_tensor, array_int_tensor.bitwise_and(mask))
        else:
            matmul_result = torch.matmul(normal_int_tensor, array_int_tensor.bitwise_and(mask))

    if array_tensor_type == TensorType.Ref:
        matmul_result = matmul_result[:, 0:-1] - matmul_result[:, -1].unsqueeze(0).t()
    elif array_tensor_type == TensorType.Normal:
        # todo: need unit test
        # do nothing
        pass
    else:
        raise Exception("We don't implement this tensor_type!", array_tensor_type)

    if matmul_result_para is None:
        matmul_result_para = creat_quantization_para(device=normal_int_tensor.device,
                                                     s=normal_s + array_s, tensor_type=TensorType.Normal)
    else:
        int_matmul_result_para = to_int(matmul_result_para)
        assert (int_matmul_result_para[2] == TensorType.Normal.value)
        int_matmul_result_para[0] = normal_s + array_s

    set_bit_width_([matmul_result, matmul_result_para], data_flow_bit_width)
    return to_float(matmul_result), to_float(matmul_result_para)


def quantization_t_matmul(normal_tensor_list: list, array_tensor_list: list, matmul_result_para: Tensor = None,
                          mask_bit_width: int = abandon_bit_with) -> [Tensor, Tensor]:
    normal_tensor, normal_para = normal_tensor_list

    return quantization_matmul([normal_tensor.t(), normal_para], array_tensor_list, matmul_result_para, mask_bit_width)


def quantization_tensor_less(float_tensor_list: list, a: float) -> torch.BoolTensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    if tensor_type == TensorType.Normal:
        int_a = round(a / pow(2, s))

        return int_tensor < int_a
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)


def mul_num(float_tensor_list: list, alpha: float, alpha_bit_width: int = data_flow_bit_width) -> [Tensor, Tensor]:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    # temp_bits = get_effective_bit_width(float_tensor_list)

    alpha_s = get_fixed_point_position(abs(alpha), alpha_bit_width)
    alpha_resolution = pow(2, alpha_s)
    alpha_int = round(alpha / alpha_resolution)
    # print(f"alpha_int: {alpha_int} alpha_s: {alpha_s} temp_bits: {temp_bits}")

    if tensor_type == TensorType.Normal:
        mul_num_int_tensor = int_tensor.mul(alpha_int)
        mul_num_s = s + alpha_s
        mul_num_para = creat_quantization_para(device=int_tensor.device, s=mul_num_s, tensor_type=TensorType.Normal)
    elif tensor_type == TensorType.Ref:
        data_part = int_tensor[..., 0:-1]
        ref_part = int_tensor[..., -1].unsqueeze(0).t()
        mul_num_int_tensor = (data_part - ref_part).mul(alpha_int)
        mul_num_s = s + alpha_s
        mul_num_para = creat_quantization_para(device=int_tensor.device, s=mul_num_s, tensor_type=TensorType.Normal)
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)

    set_bit_width_([mul_num_int_tensor, mul_num_para], data_flow_bit_width)
    return [to_float(mul_num_int_tensor), to_float(mul_num_para)]


def add_alpha_tensor_(source_tensor_list: list, add_tensor_list: list, alpha: float = None,
                      alpha_bit_width: int = data_flow_bit_width, mode: RightShiftMode = RightShiftMode.Round,
                      strategy: WeightUpdateStrategy = WeightUpdateStrategy.DynamicRange):
    source_int_tensor, source_quantization_para = parse_tensor_list_to_int(source_tensor_list)
    source_s, source_bit_width, source_tensor_type = parse_quantization_para(source_quantization_para)

    if alpha is None:
        mul_num_int_tensor, mul_num_quantization_para = parse_tensor_list_to_int(add_tensor_list)
    else:
        mul_num_int_tensor, mul_num_quantization_para = \
            parse_tensor_list_to_int(mul_num(add_tensor_list, alpha, alpha_bit_width))

    mul_num_s, mul_num_bit_width, mul_num_tensor_type = parse_quantization_para(mul_num_quantization_para)

    shift = source_s - mul_num_s
    if source_tensor_type == TensorType.Normal:
        if shift > 0:
            assert (source_bit_width + shift < system_bit_width)  # one bit for sign bit
            source_int_tensor.__ilshift__(shift).add_(mul_num_int_tensor)

            if mode == RightShiftMode.Abandon:
                source_int_tensor.__irshift__(shift)
            elif mode == RightShiftMode.Round:
                round_rshift_(source_int_tensor, shift)
            else:
                raise Exception("We don't support this right shift mode!", mode)
        else:
            assert (mul_num_bit_width - shift < system_bit_width)  # one bit for sign bit
            mul_num_int_tensor.__ilshift__(-shift)
            source_int_tensor.add_(mul_num_int_tensor)

        if strategy == WeightUpdateStrategy.StaticRange:
            neg_levels = pow_2_n(source_bit_width - 1)
            pos_levels = neg_levels - 1
            source_int_tensor[source_int_tensor < -neg_levels] = -neg_levels
            source_int_tensor[source_int_tensor > pos_levels] = pos_levels
        elif strategy == WeightUpdateStrategy.DynamicRange:
            set_bit_width_([source_int_tensor, source_quantization_para], source_bit_width)
        else:
            raise Exception("We don't support this weight update strategy!", strategy)
    elif source_tensor_type == TensorType.Ref:
        if shift > 0:
            # todo: modify here
            # implementation 1
            if mode == RightShiftMode.Abandon:
                source_int_tensor.add_(mul_num_int_tensor.__rshift__(shift))
            elif mode == RightShiftMode.Round:
                source_int_tensor.add_(round_rshift(mul_num_int_tensor, shift))
            else:
                raise Exception("We don't support this right shift mode!", mode)
            # implementation 2
            # if source_bit_width + shift >= system_bit_width:
            #     raise Exception("shift bits err! shift:", shift, source_s, mul_num_s)
            # assert (source_bit_width + shift < system_bit_width)  # one bit for sign bit
            # neg_levels = source_int_tensor[0, -1].item()
            # data_part = source_int_tensor[:, 0:-1]
            # data_part.sub_(neg_levels).__ilshift__(shift)
            # source_int_tensor.add_(mul_num_int_tensor)
            #
            # if mode == RightShiftMode.Abandon:
            #     data_part.__irshift__(shift)
            # elif mode == RightShiftMode.Round:
            #     round_rshift_(data_part, shift)
            # else:
            #     raise Exception("We don't support this right shift mode!", mode)
            #
            # data_part.add_(neg_levels)
        else:
            assert (mul_num_bit_width - shift < system_bit_width)  # one bit for sign bit
            mul_num_int_tensor.__ilshift__(-shift)
            source_int_tensor.add_(mul_num_int_tensor)

        if strategy == WeightUpdateStrategy.StaticRange:
            max_int_number = pow_2_n(source_bit_width) - 1
            source_int_tensor[source_int_tensor < 0] = 0
            source_int_tensor[source_int_tensor > max_int_number] = max_int_number
        elif strategy == WeightUpdateStrategy.DynamicRange:
            set_bit_width_([source_int_tensor, source_quantization_para], source_bit_width)
        else:
            raise Exception("We don't support this weight update strategy!", strategy)
    else:
        raise Exception("We don't implement this source_tensor_type!", source_tensor_type)
