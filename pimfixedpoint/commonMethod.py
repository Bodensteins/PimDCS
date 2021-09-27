import math
import sys


def get_fixed_point_position(max_abs: float, bit_width: int) -> int:
    return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))


def pow_2_n(n: int) -> int:
    return 1 << n


def print_call_abstract_method_info():
    print(f'Error: {sys._getframe().f_code.co_name} is an abstract method!')


def print_call_incomplete_method_info():
    print(f'Error: {sys._getframe().f_code.co_name} is an incomplete method!')


def get_pos_bit_width(pos_int: int):
    return math.ceil(math.log2(pos_int + 1)) + 1


def get_neg_bit_width(neg_int: int):
    return math.ceil(math.log2(-neg_int)) + 1
