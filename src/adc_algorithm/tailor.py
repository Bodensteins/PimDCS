import math

def tailor_sar(data_dict_list, resolution, weight_bit_width, input_bit_width, noise_max):
    noise_alloc = noise_max / (weight_bit_width * input_bit_width)
    noise_bit = math.floor(math.log2(noise_alloc))
    
    total_cmp_times = 0
    for n, data_dict in enumerate(data_dict_list):
        i = n // weight_bit_width
        j = n - i * weight_bit_width

        redundant_bit = noise_bit - i - j
        if redundant_bit > resolution:
            redundant_bit = resolution
        elif redundant_bit < 0:
            redundant_bit = 0

        # print(p, math.log2(noise_max), noise_bit, i, j, redundant_bit)

        total_cmp_times += sum(data_dict.values()) * (resolution - redundant_bit)

    return total_cmp_times


def tailor_slice_sar(data_dict, resolution, noise_alloc, c, w):
    noise_bit = math.floor(math.log2(noise_alloc))
    redundant_bit = noise_bit - c- w
    if redundant_bit > resolution:
            redundant_bit = resolution
    elif redundant_bit < 0:
        redundant_bit = 0
    total_cmp_times = sum(data_dict.values()) * (resolution - redundant_bit)
    return total_cmp_times