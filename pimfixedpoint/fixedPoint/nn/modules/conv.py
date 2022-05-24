from typing import List
import torch
import math
from torch import Tensor
import torch.nn.functional
from .. import functional as fpF
from ..fixedPointArithmetic import creat_quantization_para, quantization_tensor
from torch.nn import Module
from ..commonConst import TensorType, torch_float
from typing import Tuple


class Conv2d(Module):
    def __init__(self, input_shape: List, output_chs: int, kernel_size: Tuple, batch_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, inputBits: int = 16, weightBits: int = 16,
                 gradOutputBits: int = 16, static_tensor_mode: str = "NormalTensor", quantizerMode: str = "",
                 bias: bool = True, device: torch.device = torch.device("cpu")):
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

        # self.inputArr = None

        self.hasBias = bias
        if self.hasBias:
            self.m += 1

        self.inputBits, self.weightBits, self.gradOutputBits = inputBits, weightBits, gradOutputBits
        self.quantizerMode = quantizerMode

        self.device = device

        # this bit_width is not used in fact.
        self.pim_delta_qweight_t_cfg = torch.nn.Parameter(
            creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))

        if static_tensor_mode == "NormalTensor":
            self.pim_inArr_cfg = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal,
                                                         device=device)
            self.pim_wArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))
            self.pim_wtArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))

            input_array_row = batch_size * self.output_size[0] * self.output_size[1]
            self.inputArr = torch.empty([input_array_row, self.m], dtype=torch_float, device=device)
        else:
            raise Exception("We don't implement this tensor mode!", static_tensor_mode)

        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty([self.output_chs, self.input_chs, self.kernel_h, self.kernel_w], device=self.device)
        torch.nn.init.kaiming_uniform_(temp_weight, math.sqrt(5))

        temp_weight = temp_weight.reshape(self.output_chs, -1).T

        if self.hasBias:
            temp_bias = torch.empty(self.output_chs, device=self.device)
            _, fan_in = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight)  # our weight is different
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(0)
            temp_weight = torch.cat((temp_weight, temp_bias), 0)

        self.pim_weight = torch.nn.Parameter(temp_weight.clone().detach().requires_grad_())
        self.pim_wArr = torch.nn.Parameter(
            quantization_tensor(self.pim_wArr_cfg, temp_weight))
        self.pim_wtArr = torch.nn.Parameter(
            quantization_tensor(self.pim_wtArr_cfg, temp_weight.t()))

    def forward(self, _input: Tensor):
        # reshaple qinput to the matrix-shape
        # Here, for simplicity, we use dequan & unfold & quan to simulate fixed-point unfold.
        batch_size = _input.size()[0]

        _input = torch.nn.functional.unfold(_input, self.kernel_size, dilation=self.dilation, padding=self.padding,
                                              stride=self.stride)
        _input = _input.transpose(1, 2).reshape(-1, self.kernel_size_all)
        fp_input, qinput_config = fpF.quan.apply(_input, self.inputBits)

        # re-use pimlinerfunction to get the answer
        if self.training:
            qoutput, qoutput_config, _ = \
                fpF.linear.apply(fp_input, qinput_config, self.inputArr, self.pim_inArr_cfg, self.pim_wArr,
                                        self.pim_wArr_cfg, self.pim_wtArr, self.pim_wtArr_cfg,
                                        self.pim_delta_qweight_t_cfg,
                                        self.inputBits, self.gradOutputBits, self.hasBias, None, self.pim_weight)
        else:
            qoutput, qoutput_config, _ = \
                fpF.linear.apply(fp_input, qinput_config, None, self.pim_inArr_cfg, self.pim_wArr,
                                        self.pim_wArr_cfg, self.pim_wtArr, self.pim_wtArr_cfg,
                                        self.pim_delta_qweight_t_cfg,
                                        self.inputBits, self.gradOutputBits, self.hasBias, None, self.pim_weight)

        # reshape the qoutput to the conv-shape
        # Here, for simpicity, we use dequan & fold & quan to simulate fixed-point fold
        output = fpF.dequan.apply(qoutput, qoutput_config, None, None)
        output = output.reshape(batch_size, -1, self.output_chs).transpose(1, 2)
        output = torch.nn.functional.fold(output, self.output_size, (1, 1))

        return output
