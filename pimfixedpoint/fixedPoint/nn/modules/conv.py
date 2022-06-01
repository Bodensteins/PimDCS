from typing import List
import torch
import math
from torch import Tensor
import torch.nn.functional
from torch.nn import init
from .. import functional as fpF
from ..fixedPointArithmetic import creat_quantization_para, quantization_tensor
from torch.nn import Module
from ..commonConst import TensorType
from typing import Tuple


class Conv2d(Module):
    def __init__(self, input_shape: List, output_chs: int, kernel_size: Tuple, stride: int = 1, padding: int = 0,
                 dilation: int = 1, inputBits: int = 16, weightBits: int = 16, gradOutputBits: int = 16,
                 static_tensor_mode: str = "NormalTensor", quantizerMode: str = "", bias: bool = True):
        # todo: don't have some para
        super().__init__()
        self.output_chs = output_chs
        self.input_shape = input_shape
        self.kernel_size = kernel_size

        self.input_chs = self.input_shape[0]
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        self.kernel_h = self.kernel_size[0]
        self.kernel_w = self.kernel_size[1]

        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.dilation = (dilation, dilation) if type(dilation) == int else dilation

        self.m = self.kernel_h * self.kernel_w * self.input_chs
        self.kernel_size_all = self.m
        self.n = self.output_chs
        self.output_size = (math.floor((self.input_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_h - 1) - 1)
                                       / self.stride[0] + 1),
                            math.floor((self.input_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_w - 1) - 1)
                                       / self.stride[1] + 1))

        self.hasBias = bias
        if self.hasBias:
            self.m += 1

        self.inputBits, self.weightBits, self.gradOutputBits = inputBits, weightBits, gradOutputBits
        self.quantizerMode = quantizerMode

        self.fp_weight = None

        # this bit_width is not used in fact.
        # self.fp_delta_weight_cfg = torch.nn.Parameter(
        #     creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal))

        if static_tensor_mode == "NormalTensor":
            self.fp_weight_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal))
        else:
            raise Exception("We don't implement this tensor mode!", static_tensor_mode)

        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty([self.output_chs, self.input_chs, self.kernel_h, self.kernel_w])
        init.kaiming_uniform_(temp_weight, math.sqrt(5))

        temp_weight = temp_weight.reshape(self.output_chs, -1)

        if self.hasBias:
            temp_bias = torch.empty(self.output_chs)
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(1)
            temp_weight = torch.cat((temp_weight, temp_bias), 1)

        self.fp_weight = torch.nn.Parameter(quantization_tensor(self.fp_weight_cfg, temp_weight))

    def forward(self, _input: Tensor):
        # reshape qinput to the matrix-shape
        # Here, for simplicity, we use dequan & unfold & quan to simulate fixed-point unfold.
        batch_size = _input.size()[0]

        _input = torch.nn.functional.unfold(_input, self.kernel_size, dilation=self.dilation, padding=self.padding,
                                            stride=self.stride)
        _input = _input.transpose(1, 2).reshape(-1, self.kernel_size_all)
        fp_input, qinput_config = fpF.quan.apply(_input, self.inputBits)

        # re-use linear function to get the answer
        qoutput, qoutput_config = fpF.linear.apply(fp_input, qinput_config, self.fp_weight, self.fp_weight_cfg,
                                                   self.hasBias, self.inputBits, self.gradOutputBits)

        # reshape the output to the conv-shape
        # Here, for simplicity, we use dequan & fold & quan to simulate fixed-point fold
        output = fpF.dequan.apply(qoutput, qoutput_config, None, None)
        output = output.reshape(batch_size, -1, self.output_chs).transpose(1, 2)
        output = torch.nn.functional.fold(output, self.output_size, (1, 1))

        return output
