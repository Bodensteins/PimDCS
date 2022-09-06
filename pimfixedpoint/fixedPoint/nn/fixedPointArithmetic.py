import torch
from torch import Tensor
import math
from .commonConst import torch_int, torch_float, system_bit_width, data_flow_bit_width, TensorType, \
    RightShiftMode, WeightUpdateStrategy


def to_float(tensor):
    return tensor.view(dtype=torch_float)


def to_int(tensor):
    return tensor.detach().view(dtype=torch_int)


def _matmul_int_cuda(int_tensor1, int_tensor2):
    """matmul of tensor in cuda, int_tensor matmul int_tensor is not supported in pytorch now(2021/11/6).

        :param int_tensor1: a int tensor.
        :param int_tensor2: another int tensor.

        :return torch.matmul(int_tensor1, int_tensor2), return type:torch_int.
    """
    if not int_tensor1.is_cuda or not int_tensor2.is_cuda:
        raise Exception("tensor is not cuda, tensor 1: " + str(int_tensor1.is_cuda) +
                        ", tensor 2: " + str(int_tensor2.is_cuda))

    float_tensor1 = int_tensor1.to(dtype=torch_float)
    float_tensor2 = int_tensor2.to(dtype=torch_float)
    mul_result = float_tensor1.matmul(float_tensor2)

    return mul_result.to(dtype=torch_int)


def _get_fixed_point_position(max_abs: float, bit_width: int, tensor_type=TensorType.Normal) -> int:
    if tensor_type == TensorType.Normal:
        return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))
    elif tensor_type == TensorType.PN:
        return math.ceil(math.log2(max_abs / ((1 << bit_width) - 1)))
    else:
        raise Exception("Unknown tensor type : " + str(tensor_type.value))


def _get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def _get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1


def _round_rshift_(int_tensor, shift: int):
    if shift <= 0 or shift > system_bit_width - 1:
        raise Exception("Inappropriate shift value: " + str(shift))
    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    int_tensor.add_(round_bit).__irshift__(shift)


def _round_rshift(int_tensor, shift: int):
    if shift <= 0:
        raise Exception("Inappropriate shift value: " + str(shift))

    if shift > system_bit_width - 1:
        raise Exception("warning! right shift too many bits: " + str(shift))
        return torch.zeros_like(int_tensor)

    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    return int_tensor.add(round_bit).__rshift__(shift)


def _round_to_nearest_even_rshift(int_tensor, shift: int):
    # IEEE754 round
    if shift <= 0 or shift > system_bit_width - 1:
        raise Exception("Inappropriate shift value: " + str(shift))
    const_round_sticky = (1 << shift) - 1
    const_round = 1 << (shift - 1)
    const_guard = 1 << shift
    round_bit = int_tensor.bitwise_and(const_round_sticky)
    add_one = (round_bit > const_round).\
        __or__((round_bit == const_round).__and__(int_tensor.bitwise_and(const_guard) != 0))
    return int_tensor.__rshift__(shift).add(add_one)


def _round_to_nearest_even_rshift_(int_tensor, shift: int):
    # IEEE754 round
    if shift <= 0 or shift > system_bit_width - 1:
        raise Exception("Inappropriate shift value: " + str(shift))
    const_round_sticky = (1 << shift) - 1
    const_round = 1 << (shift - 1)
    const_guard = 1 << shift
    round_bit = int_tensor.bitwise_and(const_round_sticky)
    add_one = (round_bit > const_round).\
        __or__((round_bit == const_round).__and__(int_tensor.bitwise_and(const_guard) != 0))
    return int_tensor.__irshift__(shift).add_(add_one)


def _parse_quantization_para(quantization_para):
    """"
    input: quantization parameters, should be int Tensor
    return: quantization para. {s, bit_width, tensor_type(Normal or ref or PN)}
    """
    quantization_para = to_int(quantization_para)
    s = quantization_para[0].item()
    tensor_type = TensorType(quantization_para[1].item())

    return s, tensor_type


def _parse_tensor_tuple_to_int(fp_tensor_tuple: tuple):
    int_tensor = to_int(fp_tensor_tuple[0])
    quantization_para = to_int(fp_tensor_tuple[1])

    return int_tensor, quantization_para


def _creat_quantization_para(s: int = None, tensor_type: TensorType = None, device=None):
    """
    [0]: s [1]: tensor_type
    :param s:
    :param tensor_type:
    :param device:
    :return: quantization_para
    """
    quantization_para = torch.empty(2, dtype=torch_int, device=device)
    if s is not None:
        quantization_para[0] = s
    # if bit_width is not None:
    #     quantization_para[1] = bit_width
    if tensor_type is not None:
        quantization_para[1] = tensor_type.value

    return quantization_para


def quantization_tensor(tensor: Tensor, bit_width: int, tensor_type: TensorType):
    """
    :param tensor_type:
    :param bit_width:
    :param tensor: Float type tensor, that is the real number for you to quantize
    :return: int tensor, int para.
    """
    max_abs_value = tensor.abs().max().item()

    s = _get_fixed_point_position(max_abs_value, bit_width, tensor_type)
    quantization_para = \
        _creat_quantization_para(s=s, tensor_type=tensor_type, device=tensor.device)

    resolution = pow(2, s)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        int_tensor = tensor.div(resolution).round().to(torch_int)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))

    return int_tensor, quantization_para


def de_quantization(fp_tensor_tuple: tuple) -> Tensor:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    resolution = pow(2, s)
    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        return int_tensor.to(torch_float).mul(resolution)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))


# def get_element_wise_effective_bit_width(int_tensor, tensor_type):
#     assert (int_tensor.dtype == torch_int)
#     if tensor_type is TensorType.Ref:
#         bias = int_tensor[0, -1].item()
#         int_tensor = int_tensor.sub(bias)
#         int_tensor = int_tensor[:, 0:-1]
#
#     bit_width_dict = {}
#     for element in int_tensor.flatten():
#         element = element.item()
#         while element != 0 and element % 2 == 0:
#             element >>= 1
#
#         if element < 0:
#             bit_width = math.ceil(math.log2(-element))
#         else:
#             bit_width = math.ceil(math.log2(element + 1))
#         if bit_width == 16:
#             print(f"element: {element}")
#         bit_width += 1
#         if bit_width in bit_width_dict:
#             bit_width_dict[bit_width] = bit_width_dict[bit_width] + 1
#         else:
#             bit_width_dict[bit_width] = 1
#
#     return bit_width_dict


def _get_effective_bit_width(fp_tensor_tuple: tuple):
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)
    _, tensor_type = _parse_quantization_para(quantization_para)

    bit_width = 1

    if tensor_type == TensorType.Normal:
        max_int = int_tensor.max().item()
        min_int = int_tensor.min().item()
        pos_bit_width = 0
        neg_bit_width = 0

        if max_int > 0:
            pos_bit_width = _get_pos_bit_width(max_int)
        if min_int < 0:
            neg_bit_width = _get_neg_bit_width(min_int)

        bit_width = max(pos_bit_width, neg_bit_width, bit_width)
    elif tensor_type == TensorType.PN:
        max_abs_int = int_tensor.abs().max().item()
        max_abs_bit_width = math.ceil(math.log2(max_abs_int + 1))

        bit_width = max(bit_width, max_abs_bit_width)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))

    return bit_width


def set_bit_width_(fp_tensor_tuple: tuple, new_bit_width: int, mode: RightShiftMode = RightShiftMode.Round):
    effective_bit_width = _get_effective_bit_width(fp_tensor_tuple)

    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    if new_bit_width < 1:
        raise Exception("Illegal new bit width: " + str(new_bit_width))

    if new_bit_width < effective_bit_width:
        quantization_para[0] = s + effective_bit_width - new_bit_width

        if mode == RightShiftMode.Abandon:
            int_tensor.__irshift__(effective_bit_width - new_bit_width)
        elif mode == RightShiftMode.Round:
            _round_rshift_(int_tensor, effective_bit_width - new_bit_width)
        else:
            raise Exception("We don't support this right shift mode!", mode)


def get_clone(fp_tensor_tuple: tuple, new_bit_width: int, mode: RightShiftMode = RightShiftMode.Round):
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)

    clone_tensor = int_tensor.clone()
    clone_para = quantization_para.clone()

    set_bit_width_((clone_tensor, clone_para), new_bit_width, mode=mode)

    return to_float(clone_tensor), to_float(clone_para)


def add_additional_col_of_one(fp_tensor_tuple: tuple) -> Tensor:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    if tensor_type != TensorType.Normal:
        raise Exception("Illegal tensor type: " + str(tensor_type.value))

    decimal_part_bit_width = -s

    if decimal_part_bit_width < 0:
        print("decimal part bit width is too short: " + str(decimal_part_bit_width))

        int_one = 0
    else:
        if decimal_part_bit_width > data_flow_bit_width - 2:
            right_shift = decimal_part_bit_width - (data_flow_bit_width - 2)
            _round_rshift_(int_tensor, right_shift)
            quantization_para = quantization_para.clone()  # to avoid inplace op
            quantization_para[0] += right_shift
            decimal_part_bit_width = data_flow_bit_width - 2

        int_one = 1 << decimal_part_bit_width

    full_one_col = torch.full([int_tensor.size()[0], 1], int_one, dtype=torch_int, device=int_tensor.device)
    int_tensor = torch.cat((int_tensor, full_one_col), 1)

    return int_tensor, quantization_para


def fixed_point_matmul(input_tensor_tuple: tuple, other_tensor_tuple: tuple,
                       result_bit_width=data_flow_bit_width) -> [Tensor, Tensor]:
    """a normal tensor matmul a static tensor.
        :param result_bit_width: bit width of result
        :param input_tensor_tuple: a normal tensor list.
        :param other_tensor_tuple: a static tensor list.
        """
    normal_int_tensor, normal_quantization_para = _parse_tensor_tuple_to_int(input_tensor_tuple)
    normal_s, normal_tensor_type = _parse_quantization_para(normal_quantization_para)
    static_int_tensor, static_quantization_para = _parse_tensor_tuple_to_int(other_tensor_tuple)
    static_s, static_tensor_type = _parse_quantization_para(static_quantization_para)

    if normal_tensor_type != TensorType.Normal:
        raise Exception("Illegal normal tensor type: " + str(normal_tensor_type.value))

    if normal_int_tensor.is_cuda:
        matmul_result = _matmul_int_cuda(normal_int_tensor, static_int_tensor)
    else:
        matmul_result = torch.matmul(normal_int_tensor, static_int_tensor)

    matmul_result_para = _creat_quantization_para(device=normal_int_tensor.device, s=normal_s + static_s,
                                                  tensor_type=TensorType.Normal)

    set_bit_width_((matmul_result, matmul_result_para), result_bit_width)
    return matmul_result, matmul_result_para


def fixed_point_less(tensor_tuple: tuple, a: float) -> torch.BoolTensor:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    if tensor_type == TensorType.Normal:
        int_a = round(a / pow(2, s))

        return int_tensor < int_a
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)


def fixed_point_mul(tensor_tuple: tuple, alpha: float, alpha_bit_width: int = data_flow_bit_width) -> [Tensor, Tensor]:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    alpha_s = _get_fixed_point_position(abs(alpha), alpha_bit_width)
    alpha_resolution = pow(2, alpha_s)
    alpha_int = round(alpha / alpha_resolution)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        mul_num_int_tensor = int_tensor.mul(alpha_int)
        mul_num_s = s + alpha_s
        mul_num_para = _creat_quantization_para(device=int_tensor.device, s=mul_num_s, tensor_type=TensorType.Normal)
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)

    set_bit_width_((mul_num_int_tensor, mul_num_para), data_flow_bit_width)
    return mul_num_int_tensor, mul_num_para


def fixed_point_mul_(tensor_tuple: tuple, alpha: float, alpha_bit_width: int = data_flow_bit_width) -> [Tensor, Tensor]:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(tensor_tuple)
    _, tensor_type = _parse_quantization_para(quantization_para)

    alpha_s = _get_fixed_point_position(abs(alpha), alpha_bit_width)
    alpha_resolution = pow(2, alpha_s)
    alpha_int = round(alpha / alpha_resolution)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        int_tensor.mul_(alpha_int)
        quantization_para[0] += alpha_s
    else:
        raise Exception("We don't implement this tensor_type!", tensor_type)

    set_bit_width_(tensor_tuple, data_flow_bit_width)
    return tensor_tuple


# todo: modify its implementation, have bugs
def fixed_point_add_(source_tensor_tuple: tuple, other_tensor_tuple: tuple, source_bit_width: int = data_flow_bit_width,
                     alpha: float = None, alpha_bit_width: int = data_flow_bit_width,
                     mode: RightShiftMode = RightShiftMode.Round,
                     strategy: WeightUpdateStrategy = WeightUpdateStrategy.DynamicRange):
    source_int_tensor, source_quantization_para = _parse_tensor_tuple_to_int(source_tensor_tuple)
    source_s, source_tensor_type = _parse_quantization_para(source_quantization_para)

    if alpha is None:
        mul_num_int_tensor, mul_num_quantization_para = _parse_tensor_tuple_to_int(other_tensor_tuple)
    else:
        mul_num_int_tensor, mul_num_quantization_para = \
            _parse_tensor_tuple_to_int(fixed_point_mul(other_tensor_tuple, alpha, alpha_bit_width))

    mul_num_s, mul_num_tensor_type = _parse_quantization_para(mul_num_quantization_para)

    shift = source_s - mul_num_s
    if source_tensor_type == TensorType.Normal or TensorType.PN:
        if shift > 0:
            # todo: modify here
            if mode == RightShiftMode.Abandon:
                source_int_tensor.add_(mul_num_int_tensor.__rshift__(shift))
            elif mode == RightShiftMode.Round:
                source_int_tensor.add_(_round_rshift(mul_num_int_tensor, shift))
            else:
                raise Exception("We don't support this right shift mode!", mode)
        else:
            mul_num_bit_width = _get_effective_bit_width((mul_num_int_tensor, mul_num_quantization_para))
            if mul_num_bit_width - shift > system_bit_width:
                raise Exception("left shift too many bits, mul_num_bit_width: " + str(mul_num_bit_width) +
                                ", left shift: " + str(-shift))
            mul_num_int_tensor.__ilshift__(-shift)
            source_int_tensor.add_(mul_num_int_tensor)

        if strategy == WeightUpdateStrategy.StaticRange:
            neg_levels = 1 << (source_bit_width - 1)
            pos_levels = neg_levels - 1
            source_int_tensor[source_int_tensor < -neg_levels] = -neg_levels
            source_int_tensor[source_int_tensor > pos_levels] = pos_levels
        elif strategy == WeightUpdateStrategy.DynamicRange:
            set_bit_width_((source_int_tensor, source_quantization_para), source_bit_width)
        else:
            raise Exception("We don't support this weight update strategy!", strategy)
    else:
        raise Exception("We don't implement this source_tensor_type!", source_tensor_type)


def fixed_point_copy_(source_tensor_tuple: tuple, other_tensor_tuple: tuple):
    source_int_tensor, source_quantization_para = _parse_tensor_tuple_to_int(source_tensor_tuple)
    _, source_tensor_type = _parse_quantization_para(source_quantization_para)

    other_int_tensor, other_quantization_para = _parse_tensor_tuple_to_int(other_tensor_tuple)
    other_s, other_tensor_type = _parse_quantization_para(other_quantization_para)

    if source_int_tensor.size() != other_int_tensor.size():
        raise Exception("unequal tensor size, source: " + str(source_int_tensor.size())
                        + ", but other: " + str(other_int_tensor.size()))

    if source_tensor_type != other_tensor_type:
        raise Exception("inconsistent tensor type, source: " + str(source_tensor_type)
                        + ", but other: " + str(other_tensor_type))

    source_int_tensor.copy_(other_int_tensor)
    source_quantization_para[0] = other_s
