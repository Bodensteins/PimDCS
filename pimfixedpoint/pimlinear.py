from os import stat
import torch
import math
from torch.autograd import Function, grad

from quantization import NormalTensor, ArrayTensor, RefTensor


class PimLinearFunction(Function):
    @staticmethod
    def forward(ctx, qinput: NormalTensor, inputArr: ArrayTensor, weight: ArrayTensor, weight_t: ArrayTensor, inputBits: int, gradOutputBits: int, hasBias: bool):
        if hasBias:
            qinput.add_additional_one()

        qinput.change_bit_width(inputBits)
        inputArr.write_normal(qinput)
        qoutput = qinput.matmul_array(weight)

        ctx.inputArr = inputArr
        ctx.weight_t = weight_t
        ctx.hasBias = hasBias 
        ctx.gradOutputBits = gradOutputBits

        return qoutput 

    @staticmethod
    def backward(ctx, qgrad_output: NormalTensor):
        inputArr = ctx.inputArr
        weight_t = ctx.weight_t
        hasBias = ctx.hasBias 
        qgrad_output_bits = ctx.gradOutputBits

        qgrad_output.change_bit_width(qgrad_output_bits)
        
        qgrad_input = qgrad_output.matmul_array(weight_t)
        if hasBias:
            qgrad_input.remove_additional_one()
        delta_weight_t = qgrad_output.t_matmul_array(inputArr)

        return qgrad_input, None, None, delta_weight_t, None 


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

        if arrayMode == "RefTensor":
            self.wArr = RefTensor(weightBits)
            self.wtArr = RefTensor(weightBits)
            self.inputArr = RefTensor(weightBits)
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

        self.wArr.quantization(temp_weight, max(self.maxValue, temp_weight.abs().max()))
        self.wtArr.quantization(temp_weight.t(), max(self.maxValue, temp_weight.abs().max()))

    def forward(self, qinput: NormalTensor):
        qoutput = PimLinearFunction.apply(qinput, self.inputArr, self.wArr, self.wtArr, self.inputBits, self.gradOutputBits, self.hasBias)
        return qoutput


class DeQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: NormalTensor, x: NormalTensor, bit: int, quantizerMode: str):
        ctx.x = x
        if quantizerMode == "dynamic":
            ctx.bit = qinput.bit_width
        else:
            ctx.bit = bit
        return qinput.de_quantization()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x = ctx.x
        x.quantization(grad_output, grad_output.abs().max(), ctx.bit)
        return x, None, None, None


class DeQuanLayer(torch.nn.Module):
    def __init__(self, bitWidth: int = 8, quantizerMode: str = "dynamic"):
        super().__init__()
        self.x = NormalTensor()
        self.quantizerMode = quantizerMode
        self.bit = bitWidth

    def forward(self, qinput: NormalTensor):
        return DeQuanFunction.apply(qinput, self.x, self.bit, self.quantizerMode)


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, qinput: NormalTensor):
        neg_position = qinput.fixed_tensor < 0
        ctx.neg_position = neg_position
        qinput.fixed_tensor[neg_position] = 0
        return qinput

    @staticmethod
    def backward(ctx, grad_output: NormalTensor):
        neg_position = ctx.neg_position
        grad_output[neg_position] = 0
        return grad_output


class PimRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput: NormalTensor):
        return ReluFunction.apply(qinput)


if __name__ == "__main__":
    pass
