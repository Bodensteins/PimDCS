import math


def get_fixed_point_position(max_abs: float, bit_width: int) -> int:
    return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))


# neg levels: 2^(n-1)
def neg_levels(bit_width: int):
    return 1 << (bit_width - 1)


# pos levels: 2^(n-1) - 1
def pos_levels(bit_width: int):
    return (1 << (bit_width - 1)) - 1
