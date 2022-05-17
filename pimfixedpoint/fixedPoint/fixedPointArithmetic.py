import torch
from torch import Tensor
import math

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


def matmul_int_cuda(int_tensor1, int_tensor2):
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


def get_fixed_point_position(max_abs: float, bit_width: int, tensor_type=TensorType.Normal) -> int:
    if tensor_type == TensorType.Normal:
        return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))
    elif tensor_type == TensorType.PN:
        return math.ceil(math.log2(max_abs / ((1 << bit_width) - 1)))
    else:
        raise Exception("Unknown tensor type : " + str(tensor_type.value))


def pow_2_n(n: int) -> int:
    return 1 << n


def get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1


def round_rshift_(int_tensor, shift: int):
    if shift <= 0 or shift > system_bit_width - 1:
        raise Exception("Inappropriate shift value: " + str(shift))
    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    int_tensor.add_(round_bit).__irshift__(shift)


def round_rshift(int_tensor, shift: int):
    if shift <= 0:
        raise Exception("Inappropriate shift value: " + str(shift))

    if shift > system_bit_width - 1:
        print("warning! right shift too many bits: " + str(shift))
        return torch.zeros_like(int_tensor)

    round_bit = int_tensor.bitwise_and(1 << (shift - 1))
    return int_tensor.add(round_bit).__rshift__(shift)


def round_to_nearest_even_rshift(int_tensor, shift: int):
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


def round_to_nearest_even_rshift_(int_tensor, shift: int):
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


def parse_quantization_para(quantization_para):
    """"
    input: quantization parameters, should be int Tensor
    return: quantization para. {s, bit_width, tensor_type(Normal or ref or PN)}
    """
    quantization_para = to_int(quantization_para)
    s = quantization_para[0].item()
    bit_width = quantization_para[1].item()
    tensor_type = TensorType(quantization_para[2].item())

    return s, bit_width, tensor_type


def print_quantization_info(quantization_para):
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)
    resolution = pow(2, s)

    if tensor_type == TensorType.PN:
        neg_levels = (1 << bit_width) - 1
        pos_levels = neg_levels
    elif tensor_type == TensorType.Normal:
        neg_levels = 1 << (bit_width - 1)
        pos_levels = neg_levels - 1
    else:
        raise Exception("Unknown tensor type : " + str(tensor_type.value))

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


def quantization_tensor(quantization_para: Tensor, tensor: Tensor) -> Tensor:
    """
    float-int type: int type, but shown as float
    int-float type: float type, but shown as int
    :param quantization_para: parameters for quantization, which is a float-int type tensor.
    This tensor has 3 values. [0]: s [1]: bit_width [2]: tensor_type
    :param tensor: Float type tensor, that is the real number for you to quantize
    :return: The float-int type tensor. the quantization of input tensor.
    """
    quantization_para = to_int(quantization_para)
    _, bit_width, tensor_type = parse_quantization_para(quantization_para)  # s is unknown

    max_abs_value = tensor.abs().max().item()

    s = get_fixed_point_position(max_abs_value, bit_width, tensor_type)
    quantization_para[0] = s
    resolution = pow(2, s)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        int_tensor = tensor.div(resolution).round().to(torch_int)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))

    return to_float(int_tensor)


def de_quantization(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

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


def get_effective_bit_width(float_tensor_list: list):
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    _, tensor_bit_width, tensor_type = parse_quantization_para(quantization_para)

    bit_width = 1

    if tensor_type == TensorType.Normal:
        max_int = int_tensor.max().item()
        min_int = int_tensor.min().item()
        pos_bit_width = 0
        neg_bit_width = 0

        if max_int > 0:
            pos_bit_width = get_pos_bit_width(max_int)
        if min_int < 0:
            neg_bit_width = get_neg_bit_width(min_int)

        bit_width = max(pos_bit_width, neg_bit_width, bit_width)
    elif tensor_type == TensorType.PN:
        max_abs_int = int_tensor.abs().max().item()
        max_abs_bit_width = math.ceil(math.log2(max_abs_int + 1))

        bit_width = max(bit_width, max_abs_bit_width)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))

    return bit_width


def set_bit_width_(float_tensor_list: list, new_bit_width: int, mode: RightShiftMode = RightShiftMode.Abandon):
    effective_bit_width = get_effective_bit_width(float_tensor_list)

    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, _, tensor_type = parse_quantization_para(quantization_para)

    if new_bit_width < 1:
        raise Exception("Illegal new bit width: " + str(new_bit_width))

    if new_bit_width < effective_bit_width:
        quantization_para[0] = s + effective_bit_width - new_bit_width

        if mode == RightShiftMode.Abandon:
            int_tensor.__irshift__(effective_bit_width - new_bit_width)
        elif mode == RightShiftMode.Round:
            round_rshift_(int_tensor, effective_bit_width - new_bit_width)
        else:
            raise Exception("We don't support this right shift mode!", mode)

    quantization_para[1] = new_bit_width


def add_additional_col_of_one(float_tensor_list: list) -> Tensor:
    int_tensor, quantization_para = parse_tensor_list_to_int(float_tensor_list)
    s, bit_width, tensor_type = parse_quantization_para(quantization_para)

    if tensor_type != TensorType.Normal:
        raise Exception("Illegal tensor type: " + str(tensor_type.value))

    resolution = pow(2, s)
    max_value = (pow_2_n(bit_width) - 1) * resolution

    int_one = round(1 / resolution)

    if max_value < 1:
        new_bit_width = get_pos_bit_width(int_one)
        set_bit_width_(float_tensor_list, new_bit_width)

    full_one_col = torch.full([int_tensor.size()[0], 1], int_one, dtype=torch_int, device=int_tensor.device)
    int_tensor = torch.cat((int_tensor, full_one_col), 1)

    return to_float(int_tensor)


def write_tensor_(static_tensor_list: list, data_tensor_list: list, mode: RightShiftMode = RightShiftMode.Abandon):
    """write a dynamic tensor(immediate data) into a static tensor(weight) inplace.
    :param static_tensor_list: a static tensor list.
    :param data_tensor_list: a dynamic tensor list.
    :param mode: optional, write mode.
    """
    static_int_tensor, static_tenor_para = parse_tensor_list_to_int(static_tensor_list)
    _, static_bit_width, static_tensor_type = parse_quantization_para(static_tenor_para)
    data_bit_width = get_effective_bit_width(data_tensor_list)
    data_int_tensor, data_quantization_para = parse_tensor_list_to_int(data_tensor_list)
    data_s, _, data_tensor_type = parse_quantization_para(data_quantization_para)

    if data_tensor_type != TensorType.Normal:
        raise Exception("Illegal dynamic tensor type: " + str(data_tensor_type.value))

    if static_tensor_type == TensorType.Normal or static_tensor_type == TensorType.PN:
        if static_bit_width >= data_bit_width:
            static_tenor_para[0] = data_s
        else:
            static_tenor_para[0] = data_s + data_bit_width - static_bit_width

            if mode == RightShiftMode.Abandon:
                data_int_tensor = data_int_tensor.__rshift__(data_bit_width - static_bit_width)
            elif mode == RightShiftMode.Round:
                data_int_tensor = round_rshift(data_int_tensor, data_bit_width - static_bit_width)
            else:
                raise Exception("We don't support this right shift mode: ", str(mode))

        if static_int_tensor.shape[0] == data_int_tensor.shape[0]:
            static_int_tensor.zero_().add_(data_int_tensor)
        elif static_int_tensor.shape[0] > data_int_tensor.shape[0]:
            static_int_tensor.zero_()
            static_int_tensor[0:data_int_tensor.shape[0]].add_(data_int_tensor)
        else:
            raise Exception("Illegal tensor shape, static: " + str(static_int_tensor.shape) +
                            ", but dynamic: " + str(data_int_tensor.shape))
    else:
        raise Exception("Unknown static tensor type: " + str(static_tensor_type.value))

    return to_float(static_int_tensor)


def fixed_point_matmul(normal_tensor_list: list, static_tensor_list: list, matmul_result_para: Tensor = None) \
        -> [Tensor, Tensor]:
    """a normal tensor matmul a static tensor.
        :param normal_tensor_list: a normal tensor list.
        :param static_tensor_list: a static tensor list.
        :param matmul_result_para: the result para, can be none.
        """
    normal_int_tensor, normal_quantization_para = parse_tensor_list_to_int(normal_tensor_list)
    normal_s, normal_bit_width, normal_tensor_type = parse_quantization_para(normal_quantization_para)
    static_int_tensor, static_quantization_para = parse_tensor_list_to_int(static_tensor_list)
    static_s, static_bit_width, static_tensor_type = parse_quantization_para(static_quantization_para)

    if normal_tensor_type != TensorType.Normal:
        raise Exception("Illegal normal tensor type: " + str(normal_tensor_type.value))

    if normal_int_tensor.shape[1] < static_int_tensor.shape[0]:
        #  todo: may modify the implementation
        zero_cols = torch.zeros([normal_int_tensor.shape[0], static_int_tensor.shape[0] - normal_int_tensor.shape[1]],
                                dtype=normal_int_tensor.dtype, device=normal_int_tensor.device)
        normal_int_tensor = torch.cat((normal_int_tensor, zero_cols), 1)
    elif normal_int_tensor.shape[1] > static_int_tensor.shape[0]:
        raise Exception("Illegal tensor shape, normal tensor shape: " + str(normal_int_tensor.shape) +
                        ", but static tensor shape" + str(static_int_tensor.shape))

    if normal_int_tensor.is_cuda:
        matmul_result = matmul_int_cuda(normal_int_tensor, static_int_tensor)
    else:
        matmul_result = torch.matmul(normal_int_tensor, static_int_tensor)

    if matmul_result_para is None:
        matmul_result_para = creat_quantization_para(device=normal_int_tensor.device,
                                                     s=normal_s + static_s, tensor_type=TensorType.Normal)
    else:
        int_matmul_result_para = to_int(matmul_result_para)
        if int_matmul_result_para[2] != TensorType.Normal.value:
            raise Exception("Illegal tensor type: ", str(int_matmul_result_para[2]))
        int_matmul_result_para[0] = normal_s + static_s

    set_bit_width_([matmul_result, matmul_result_para], data_flow_bit_width)
    return to_float(matmul_result), to_float(matmul_result_para)


def fixed_point_t_matmul(normal_tensor_list: list, static_tensor_list: list, matmul_result_para: Tensor = None) \
        -> [Tensor, Tensor]:
    normal_tensor, normal_para = normal_tensor_list

    return fixed_point_matmul([normal_tensor.t(), normal_para], static_tensor_list, matmul_result_para)


def fixed_point_less(float_tensor_list: list, a: float) -> torch.BoolTensor:
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

    alpha_s = get_fixed_point_position(abs(alpha), alpha_bit_width)
    alpha_resolution = pow(2, alpha_s)
    alpha_int = round(alpha / alpha_resolution)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        mul_num_int_tensor = int_tensor.mul(alpha_int)
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
    if source_tensor_type == TensorType.Normal or TensorType.PN:
        if shift > 0:
            # todo: modify here
            if mode == RightShiftMode.Abandon:
                source_int_tensor.add_(mul_num_int_tensor.__rshift__(shift))
            elif mode == RightShiftMode.Round:
                source_int_tensor.add_(round_rshift(mul_num_int_tensor, shift))
            else:
                raise Exception("We don't support this right shift mode!", mode)
        else:
            if mul_num_bit_width - shift >= system_bit_width:
                raise Exception("left shift too many bits, mul_num_bit_width: " + str(mul_num_bit_width) +
                                ", left shift: " + str(-shift))
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
    else:
        raise Exception("We don't implement this source_tensor_type!", source_tensor_type)
