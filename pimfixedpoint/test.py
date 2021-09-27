import torch
from quantization import *

row_size = 4
col_size = 4
bit_width = 8

# test ref col
# weight = torch.randn([row_size, col_size], dtype=torch.float)
# print(weight)
# max_value = weight.abs().max()
# ref_tensor = RefTensor()
# ref_tensor.quantization(weight, max_value, bit_width)
# ref_tensor.print_quantization_info()
# de_quantization_weight = ref_tensor.de_quantization()
# print(ref_tensor.fixed_tensor)
# print(de_quantization_weight)
# delta = weight - de_quantization_weight
# print(f'max abs delta: {delta.abs().max().item()}')

# test input
input_mat = torch.randn([row_size, col_size], dtype=torch.float)
print(input_mat)
max_value = input_mat.abs().max()
input_tensor = NormalTensor()
input_tensor.quantization(input_mat, max_value, bit_width)
input_tensor.print_quantization_info()
de_quantization_input = input_tensor.de_quantization()
print(input_tensor.fixed_tensor)
print(de_quantization_input)
delta = input_mat - de_quantization_input
print(f'max abs delta: {delta.abs().max().item()}')


# test P&N
# weight = torch.randn([row_size, col_size], dtype=torch.float)
# print(weight)
# max_value = weight.abs().max()
# PN_tensor = quantization.PNTensor(row_size, col_size, max_value, bit_width)
# PN_tensor.quantization_tensor(weight)
# PN_tensor.print_quantization_info()
# de_quantization_weight = PN_tensor.de_quantization()
# print(PN_tensor.pos_fixed_tensor)
# print(PN_tensor.neg_fixed_tensor)
# print(de_quantization_weight)
# delta = weight - de_quantization_weight
# print(f'max abs delta: {delta.abs().max().item()}')
