import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from fixedPoint.fixedPointArithmetic import set_bit_width_, TensorType, creat_quantization_para, quantization_tensor, \
    parse_quantization_para, de_quantization, quantization_tensor_less, to_int


class DeQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, bit: int, backBit: int, quantizerMode: str = "dynamic"):
        if quantizerMode == "dynamic":
            _, ctx.backBit, _ = parse_quantization_para(to_int(qinput_config))
        else:
            ctx.backBit = backBit

        if bit is not None:
            set_bit_width_([qinput, qinput_config], bit)

        return de_quantization([qinput, qinput_config])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        qgrad_output_config = creat_quantization_para(bit_width=ctx.backBit, tensor_type=TensorType.Normal,
                                                      device=grad_output.device)
        # print(grad_output)
        qgrad_output = quantization_tensor(qgrad_output_config, grad_output)

        return qgrad_output, qgrad_output_config, None, None, None


class DeQuanLayer(nn.Module):
    def __init__(self, bit_width: int = 16, back_bit_width: int = 16, quantization_mode: str = "dynamic"):
        super().__init__()
        self.quantizationMode = quantization_mode
        self.bit = bit_width
        self.backBit = back_bit_width

    def forward(self, qinput: Tensor, qinput_config: Tensor):
        return DeQuanFunction.apply(qinput, qinput_config, self.bit, self.backBit, self.quantizationMode)


class QuanFunction(Function):
    @staticmethod
    def forward(ctx, _input, bit_width):
        fp_output_config = creat_quantization_para(bit_width=bit_width, tensor_type=TensorType.Normal,
                                                   device=_input.device)
        fp_output = quantization_tensor(fp_output_config, _input)
        return fp_output, fp_output_config

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_config):
        return de_quantization([fp_grad_output, fp_grad_output_config]), None


class QuanLayer(nn.Module):
    def __init__(self, bit_width):
        super().__init__()
        self.bit_width = bit_width

    def forward(self, _input):
        return QuanFunction.apply(_input, self.bit_width)


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, origin_input: Tensor = None):
        neg_position = quantization_tensor_less([qinput, qinput_config], 0)
        ctx.neg_position = neg_position
        qinput[neg_position] = 0  # The zero in ieee754 is all zero as well.
        if origin_input is not None:
            origin_neg_position = origin_input < 0
            origin_input[origin_neg_position] = 0
            ctx.origin_neg_position = origin_neg_position
        return qinput, qinput_config, origin_input

    @staticmethod
    def backward(ctx, qgrad_output: Tensor, qgrad_output_config: Tensor, grad_output: Tensor):
        neg_position = ctx.neg_position
        qgrad_output[neg_position] = 0
        if grad_output is not None:
            grad_output[ctx.origin_neg_position] = 0
        return qgrad_output, qgrad_output_config, grad_output


class PimRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput: Tensor, qinput_config: Tensor, origin_input: Tensor = None):
        return ReluFunction.apply(qinput, qinput_config, origin_input)