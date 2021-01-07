# -*- coding: utf-8 -*-
import torch
import pimtorch
from pimtorch import PimLinear
from torchvision.models.vgg import vgg16
linear = PimLinear(10, 10, 1, pimtorch.PimArrayType.SimpleLogicArray)
input = torch.randn([1, 10])
output = linear.forward(input,,
print(output)
# torch.classes.load_library("cmake-build-debug-llvm/liblogic_array_interface.dylib")
# print(torch.classes.loaded_libraries)
#
# pim_arr = torch.classes.pimtorch.SimpleLogicArray(10, 10)
#
# size = pim_arr.sizes()