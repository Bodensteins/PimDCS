from os import stat
import torch
import math
from torch.autograd import Function, grad
from torch import Tensor
from quantization import NormalTensor, ArrayTensor, RefTensor

class PimLinearFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, qinputArr: Tensor, qinputArr_config: Tensor, qweight: Tensor, 
        qweight_config: Tensor, qweight_t: Tensor, qweight_t_config: Tensor, inputBits: int, gradOutputBits: int, hasBias: bool):

        if hasBias:
            add_additional_one_([qinput, qinput_config])
        
        chenge_bit_width_([qinput, qinput_config], inputBits)
        write_array_([qinputArr, qinputArr_config], [qinput, qinput_config])
        qoutput, qoutput_config = pim_matmul([qinput, qinput_config], [qweight, qweight_config])

        ctx.save_for_backward(qinputArr, qweight_t)
        ctx.qinputArr_config = qinputArr_config
        ctx.qweight_t_config = qweight_t_config

        ctx.gradOutputBits = gradOutputBits
        ctx.hasBias = hasBias

        return qoutput, qoutput_config

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        qinputArr, qweight_t = ctx.saved_tensors
        qinputArr_config = ctx.qinputArr_config
        qweight_t_config = ctx.qweight_t_config

        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits

        change_bit_width_([qgrad_output, qgrad_output_config], qgrad_output_bits)
        
        qgrad_input, qgrad_input_config = pim_matmul([qgrad_output, qgrad_output_config], [qweight_t, qweight_t_config])
        if hasBias:
            remove_additional_one_([qgrad_input, qgrad_input_config])
        delta_qweight_t = pim_t_matmul([qgrad_output, qgrad_output_config], [qinputArr, qinputArr_config])

        return qgrad_input, qgrad_input_config, None, None, None, delta_qweight_t, None, None, None, None


class PimLinear(torch.nn.Module):
    def __init__(self, m: int, n: int, inputBits: int = 8, weightBits: int = 8, gradOutputBits: int = 8,
                 arrayMode: str = "RefTensor", quantizerMode: str = "", absMaxValue: float = 1.0, hasBias: bool = True):
        super().__init__()
        if hasBias:
            m += 1
        self.m, self.n, self.hasBias, self.maxValue = m, n, hasBias, absMaxValue
        # quantizer mode dynamic, static.
        self.quantizerMode = quantizerMode 
        self.inputBits, self.gradOutputBits = inputBits, gradOutputBits
        self.wArr, self.wtArr, self.inputArr = None, None, None

        if arrayMode == "RefTensor":
            self.inputArrConfig = initRefArrayConfig(weightBits)
            self.wArrConfig = initRefArrayConfig(weightBits)
            self.wtArrConfig = initRefArrayConfig(weightBits)
        # elif arrayMode == "PNTensor":
        #    self.wArr = PNTensor(m, n, absMaxValue, bitwidth)
        #    self.wtArr = PNTensor(n, m, absMaxValue, bitwidth) 
        #    self.inputArr = PNTensor(batch_size, m, absMaxValue, bitwidth)
        else:
            print("arrayMode error!")

        self.weight_init()
    
    def weight_init(self):
        temp_weight = torch.empty(self.m, self.n)
        torch.nn.init.kaiming_uniform_(temp_weight, math.sqrt(5))

        if self.hasBias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight[0:-1])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_weight[-1], -bound, bound)

        pim_arr_quantization([self.wArr, self.wArrConfig], temp_weight, max(self.maxValue, temp_weight.abs().max()))
        pim_arr_quantization([self.wtArr, self.wtArrConfig], temp_weight.t(), max(self.maxValue, temp_weight.abs().max()))

    def forward(self, qinput: Tensor, qinput_config):
        qoutput, qoutput_config = PimLinearFunction.apply(qinput, qinput_config, self.inputArr, self.inputArrConfig, self.wArr, self.wArrConfig, self.wtArr, 
            self.wtArrConfig, self.inputBits, self.gradOutputBits, self.hasBias)

        return qoutput, qoutput_config


class DeQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, bit: int, quantizerMode: str):
        if quantizerMode == "dynamic":
            ctx.bit = get_bit(qinput_config)
        else:
            ctx.bit = bit
        return de_quantization(qinput, qinput_config)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        qgrad_output_config = initNormalConfig(ctx.bit)
        qgrad_output = torch.empty(grad_output.size())
        pim_normal_quantization([qgrad_output, qgrad_output_config], grad_output, grad_output.abs().max())
        return qgrad_output, qgrad_output_config, None, None


class DeQuanLayer(torch.nn.Module):
    def __init__(self, bitWidth: int = 8, quantizerMode: str = "dynamic"):
        super().__init__()
        self.quantizerMode = quantizerMode
        self.bit = bitWidth

    def forward(self, qinput: Tensor, qinput_config: Tensor):
        return DeQuanFunction.apply(qinput, qinput_config, self.bit, self.quantizerMode)


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor):
        neg_position = pim_le([qinput, qinput_config], 0)
        ctx.neg_position = neg_position
        qinput[neg_position] = 0 # The zero in ieee754 is all zero as well.
        return qinput, qinput_config

    @staticmethod
    def backward(ctx, qgrad_output: Tensor, qgrad_output_config: Tensor):
        neg_position = ctx.neg_position
        qgrad_output[neg_position] = 0
        return qgrad_output, qgrad_output_config


class PimRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput: Tensor, qinput_config: Tensor):
        return ReluFunction.apply(qinput, qinput_config)


if __name__ == "__main__":
    pass
