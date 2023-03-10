# import random
#
# import torch.nn.functional
#
# from fixedPoint.nn.fixedPointArithmetic import *
# import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import fixedPoint.nn.fixedPointArithmetic as fpA
import torch


def calculate_relative_error(tensor, other):
    int_source, source_cfg = tensor
    int_other, other_cfg = other
    source_s = source_cfg[0]
    other_s = other_cfg[0]
    # int_other.__ilshift__(other_s-source_s)
    int_other = int_other.__lshift__(other_s-source_s)
    delta_tensor = torch.sub(int_other, int_source).abs()
    relative_error_tensor = torch.div(delta_tensor, int_source)
    return relative_error_tensor


if __name__ == "__main__":
    test_layer_num = 16
    test_epoch_num = 25
    new_bit_width = 16

    relative_error_layer_list = []
    number_layer_list = []

    for test_layer_index in range(test_layer_num):
        for test_epoch_index in range(test_epoch_num):
            file_name = "output_test/VGG16/layer" + str(test_layer_index) + "/epoch" + str(test_epoch_index)
            output, output_cfg = torch.load(file_name)
            output_clone = output.clone()
            output_cfg_clone = output_cfg.clone()
            fpA.set_bit_width_((output_clone, output_cfg_clone), new_bit_width) # bit_width_reduction
            float_output = fpA.de_quantization((output, output_cfg))
            re_quan_output, re_quan_cfg = fpA.quantization_tensor(float_output, new_bit_width, fpA.TensorType.Normal)
            calculate_relative_error((output, output_cfg), (output_clone, output_cfg_clone))
            # cmp_result = torch.not_equal(re_quan_output, output_clone)
            # cmp_sum = torch.sum(cmp_result)

            pass
# row_size = 6
# col_Size = 4
# data_bit_width = 9
# static_bit_width = 10
# alpha = 2.3
# alpha_bit_width = 12
#
# torch.matmul()
#
#
# def print_info(int_tensor: Tensor, para: Tensor):
#     print_quantization_info(para)
#     print(to_int(int_tensor))
#     print(de_quantization((int_tensor, para)))
#
#
# data_para = _creat_quantization_para(bit_width=data_bit_width, tensor_type=TensorType.Normal)
# data_float_tensor = torch.randn([row_size, col_Size], dtype=torch.float)
# data_tensor = quantization_tensor(data_para, data_float_tensor)
#
# # print(f'normal:\n{data_float_tensor}')
# print_info(data_tensor, data_para)
#
# static_para = _creat_quantization_para(bit_width=static_bit_width, tensor_type=TensorType.Normal)
# static_float_tensor = torch.randn([row_size, col_Size], dtype=torch.float)
# static_tensor = quantization_tensor(static_para, static_float_tensor)
#
# # print(f'source:\n{static_float_tensor}')
# print_info(static_tensor, static_para)
#
# float_res = torch.add(static_float_tensor, data_float_tensor, alpha=0.01)
# print(f'float res: \n{float_res}')
#
# fixed_point_add_((static_tensor, static_para), (data_tensor, data_para), alpha=0.01)
# print_info(static_tensor, static_para)
# print(f'max delta: {float_res.sub(de_quantization((static_tensor, static_para))).abs().max()}')
#
#
# model = torchvision.models.ResNet()
# model = torchvision.models.VGG()
# torch.optim.SGD()

# array_para = creat_quantization_para(bit_width=array_bit_width, tensor_type=TensorType.Normal)
# array_float_tensor = torch.randn([row_size, col_Size], dtype=torch.float)
# array_tensor = quantization_tensor(array_para, array_float_tensor, array_float_tensor.abs().max())
#
# print('\nadd:')
# print_info(array_tensor, array_para)
#
# add_alpha_tensor_([data_tensor, data_para], [array_tensor, array_para], alpha, alpha_bit_width)
#
# print('\nadd result:')
# print_info(data_tensor, data_para)
#
# delta = data_float_tensor.add(array_float_tensor.mul(alpha)) - de_quantization([data_tensor, data_para])
# print(f'max delta: {delta.abs().max()}')


# array_para = creat_quantization_para(bit_width=array_bit_width, tensor_type=TensorType.Ref)
# float_tensor_2 = torch.randn([col_Size, row_size], dtype=torch.float)
# array_tensor = quantization_tensor(array_para, float_tensor_2, float_tensor_2.abs().max())
# s, bit_width, _ = parse_quantization_para(float_to_int(array_para))
# print_quantization_info(s, bit_width)
# print(float_to_int(array_tensor))
# print(de_quantization([array_tensor, array_para]))
#
# matmul_result, matmul_result_para = normal_t_matmul_array([int_tensor, para], [array_tensor, array_para])
# s, bit_width, _ = parse_quantization_para(float_to_int(matmul_result_para))
# print_quantization_info(s, bit_width)
# print(float_to_int(matmul_result))
# print(de_quantization([matmul_result, matmul_result_para]))
# delta = float_tensor.t().matmul(float_tensor_2) - de_quantization([matmul_result, matmul_result_para])
# print(f'max delta: {delta.abs().max()}')

# float_tensor = torch.randn([row_size, col_Size], dtype=torch.float)
# int_tensor = quantization_tensor(para, float_tensor, float_tensor.abs().max())
# s, bit_width, _ = parse_quantization_para(float_to_int(para))
# print_quantization_info(s, bit_width)
# print(float_to_int(int_tensor))
# print(de_quantization([int_tensor, para]))
#
# array_tensor = write_array([array_tensor, array_para], [int_tensor, para])
# s, bit_width, _ = parse_quantization_para(float_to_int(array_para))
# print_quantization_info(s, bit_width)
# print(float_to_int(array_tensor))
# print(de_quantization([array_tensor, array_para]))
# s, bit_width, _ = parse_quantization_para(float_to_int(para_tensor))
# print_quantization_info(s, bit_width)
# print(float_to_int(q_tensor))
# print(de_quantization([q_tensor, para_tensor]))
#
# q_tensor = add_additional_one([q_tensor, para_tensor])
# q_tensor = remove_additional_one_(q_tensor)
# s, bit_width, _ = parse_quantization_para(float_to_int(para_tensor))
# print_quantization_info(s, bit_width)
# print(float_to_int(q_tensor))
# print(de_quantization([q_tensor, para_tensor]))

# new_bit_width = 7
# change_bit_width_([q_tensor, para_tensor], new_bit_width)
#
# s, bit_width, _ = parse_quantization_para(float_to_int(para_tensor))
# print_quantization_info(s, bit_width)
# print(float_to_int(q_tensor))
#
# delta = de_quantization([q_tensor, para_tensor]) - float_tensor
# print(f'max delta: {delta.abs().max()}')


# row_size = 4
# col_size = 4
# normal_bit_width = 8
# array_bit_width = 9
# alpha = 0.01
# alpha_bit_width = 12
#
# normal_mat = torch.randn([row_size, col_size], dtype=torch.float)
# max_value = normal_mat.abs().max()
# normal_tensor = NormalTensor()
# normal_tensor.quantization(normal_mat, max_value, normal_bit_width)
# print(normal_tensor.de_quantization())
#
# array_mat = torch.randn([row_size, col_size], dtype=torch.float)
# max_value = array_mat.abs().max()
# array_tensor = RefTensor(array_bit_width)
# array_tensor.quantization(array_mat, max_value)
# print(array_tensor.de_quantization())
#
# normal_tensor.add_array_(array_tensor, alpha, alpha_bit_width)
# print(normal_tensor.de_quantization())
# normal_tensor.print_quantization_info()
# delta = normal_tensor.de_quantization() - normal_mat.add(array_mat.mul(alpha))
# print(f'max abs delta: {delta.abs().max().item()}\n')

# array_tensor.add_normal_(normal_tensor, alpha, alpha_bit_width)
# array_tensor.print_quantization_info()

# delta = array_tensor.de_quantization() - array_mat.add(normal_mat.mul(alpha))
# print(f'max abs delta: {delta.abs().max().item()}\n')


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
# print("input:")
# input_mat = torch.randn([row_size, col_size], dtype=torch.float) * 10
# print(input_mat)
# max_value = input_mat.abs().max()
# input_tensor = NormalTensor()
# input_tensor.quantization(input_mat, max_value, normal_bit_width)
# input_tensor.print_quantization_info()
# de_quantization_input = input_tensor.de_quantization()
# print(input_tensor.fixed_tensor)
# print(de_quantization_input)
# delta = input_mat - de_quantization_input
# print(f'max abs delta: {delta.abs().max().item()}\n')
#
# print("weight:")
# weight_bit_width = 8
# weight_mat = torch.randn([row_size, col_size], dtype=torch.float)
# print(weight_mat)
# max_value = weight_mat.abs().max()
# weight_tensor = RefTensor(weight_bit_width)
# weight_tensor.quantization(weight_mat, max_value)
# weight_tensor.print_quantization_info()
# de_quantization_weight = weight_tensor.de_quantization()
# print(weight_tensor.fixed_tensor)
# print(de_quantization_weight)
# delta = weight_mat - de_quantization_weight
# print(f'max abs delta: {delta.abs().max().item()}\n')
#
# print("standard mul:")
# print(input_mat.matmul(weight_mat))
#
# print("\nmul:")
# mul_res = input_tensor.matmul_array(weight_tensor)
# mul_res.print_quantization_info()
# de_quantization_mul = mul_res.de_quantization()
# print(mul_res.fixed_tensor)
# print(de_quantization_mul)
# delta = input_mat.matmul(weight_mat) - de_quantization_mul
# print(f'max abs delta: {delta.abs().max().item()}')
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
