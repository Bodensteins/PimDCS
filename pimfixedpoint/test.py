import torch
from quantization import *

row_size = 4
col_size = 4
input_bit_width = 8

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
print("input:")
input_mat = torch.randn([row_size, col_size], dtype=torch.float) * 10
print(input_mat)
max_value = input_mat.abs().max()
input_tensor = NormalTensor()
input_tensor.quantization(input_mat, max_value, input_bit_width)
input_tensor.print_quantization_info()
de_quantization_input = input_tensor.de_quantization()
print(input_tensor.fixed_tensor)
print(de_quantization_input)
delta = input_mat - de_quantization_input
print(f'max abs delta: {delta.abs().max().item()}\n')

print("weight:")
weight_bit_width = 8
weight_mat = torch.randn([row_size, col_size], dtype=torch.float)
print(weight_mat)
max_value = weight_mat.abs().max()
weight_tensor = RefTensor(weight_bit_width)
weight_tensor.quantization(weight_mat, max_value)
weight_tensor.print_quantization_info()
de_quantization_weight = weight_tensor.de_quantization()
print(weight_tensor.fixed_tensor)
print(de_quantization_weight)
delta = weight_mat - de_quantization_weight
print(f'max abs delta: {delta.abs().max().item()}\n')

print("standard mul:")
print(input_mat.matmul(weight_mat))

print("\nmul:")
mul_res = input_tensor.mul_array(weight_tensor)
mul_res.print_quantization_info()
de_quantization_mul = mul_res.de_quantization()
print(mul_res.fixed_tensor)
print(de_quantization_mul)
delta = input_mat.matmul(weight_mat) - de_quantization_mul
print(f'max abs delta: {delta.abs().max().item()}')
# print("sub:")
# weight_tensor.sub(input_tensor)
# weight_tensor.print_quantization_info()
# de_quantization_weight = weight_tensor.de_quantization()
# print(weight_tensor.fixed_tensor)
# print(de_quantization_weight)
# delta = weight_mat - input_mat - de_quantization_weight
# print(f'max abs delta: {delta.abs().max().item()}')

# input_tensor.add_additional_one()
# input_tensor.print_quantization_info()
# print(input_tensor.fixed_tensor)
# print(input_tensor.de_quantization())
#
# input_tensor.remove_additional_one()
# input_tensor.print_quantization_info()
# print(input_tensor.fixed_tensor)
# print(input_tensor.de_quantization())
#
# input_tensor.change_bit_width(input1_bit_width)
# input_tensor.print_quantization_info()
# print(input_tensor.fixed_tensor)
# print(input_tensor.de_quantization())

# input_tensor_t = input_tensor.t()
# input_tensor_t.print_quantization_info()
# print(input_tensor_t.fixed_tensor)
# print(input_tensor_t.de_quantization())
#
#
# input_tensor_t.fixed_tensor[0][0] = 6
# print(input_tensor.fixed_tensor)
# print(input_tensor_t.fixed_tensor)



# ref_bit_width = 9
# ref_tensor = RefTensor(ref_bit_width)
# ref_tensor.write(input_tensor)
# ref_tensor.print_quantization_info()
# de_quantization_ref = ref_tensor.de_quantization()
# print(ref_tensor.fixed_tensor)
# print(de_quantization_ref)
# delta = input_mat - de_quantization_ref
# print(f'max abs delta: {delta.abs().max().item()}')
#
#
# input2_bit_width = 8
# input_mat2 = torch.randn([row_size, col_size], dtype=torch.float)
# print(input_mat2)
# max_value2 = input_mat2.abs().max()
# input2_tensor = NormalTensor()
# input2_tensor.quantization(input_mat2, max_value2, input2_bit_width)
# input2_tensor.print_quantization_info()
# de_quantization_input2 = input2_tensor.de_quantization()
# print(input2_tensor.fixed_tensor)
# print(de_quantization_input2)
# delta = input_mat2 - de_quantization_input2
# print(f'max abs delta: {delta.abs().max().item()}')
#
# ref_tensor.write(input2_tensor)
# ref_tensor.print_quantization_info()
# de_quantization_ref = ref_tensor.de_quantization()
# print(ref_tensor.fixed_tensor)
# print(de_quantization_ref)
# delta = input_mat2 - de_quantization_ref
# print(f'max abs delta: {delta.abs().max().item()}')

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
