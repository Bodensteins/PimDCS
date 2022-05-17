import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from fixedPoint.fixedPointArithmetic import set_bit_width_, TensorType, creat_quantization_para, quantization_tensor, \
    parse_quantization_para, de_quantization, fixed_point_less, to_int, parse_tensor_list_to_int, torch_int, \
    to_float
import pdb

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
        pdb.set_trace()
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
        pdb.set_trace()
        return de_quantization([fp_grad_output, fp_grad_output_config]), None


class QuanLayer(nn.Module):
    def __init__(self, bit_width):
        super().__init__()
        self.bit_width = bit_width

    def forward(self, _input):
        return QuanFunction.apply(_input, self.bit_width)


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor):
        neg_position = fixed_point_less([qinput, qinput_config], 0)
        qinput[neg_position] = 0  # The zero in ieee754 is all zero as well.
        ctx.save_for_backward(neg_position)
        return qinput, qinput_config

    @staticmethod
    def backward(ctx, qgrad_output: Tensor, qgrad_output_config: Tensor):
        pdb.set_trace()
        neg_position, = ctx.saved_tensors
        qgrad_output[neg_position] = 0

        return qgrad_output, qgrad_output_config


class PimRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput: Tensor, qinput_config: Tensor):
        return ReluFunction.apply(qinput, qinput_config)


class FPDropoutFunction(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_cfg, dropout_ratio, training):
        if training:
            mask = torch.rand_like(fp_input) > dropout_ratio
            ctx.save_for_backward(mask)  # tensor should be saved in save for backward
            return fp_input.mul(mask), fp_input_cfg  # The zero in ieee754 is all zero as well.
        else:
            fp_input, _ = parse_tensor_list_to_int([fp_input, fp_input_cfg])
            fp_input = fp_input.mul(dropout_ratio).to(dtype=torch_int)

            return to_float(fp_input), fp_input_cfg

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_cfg):
        pdb.set_trace()
        mask, = ctx.saved_tensors
        return fp_grad_output.mul(mask), fp_grad_output_cfg, None, None


class FPDropout(nn.Module):
    def __init__(self, dropout_ratio=0.5):
        super(FPDropout, self).__init__()
        self.dropout_ratio = dropout_ratio

    def forward(self, fp_input, fp_input_cfg):
        return FPDropoutFunction.apply(fp_input, fp_input_cfg, self.dropout_ratio, self.training)


class MulInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
