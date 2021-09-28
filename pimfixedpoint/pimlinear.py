from os import stat
import torch
import math
from torch.autograd import Function, grad

from quantization import NormalTensor, ArrayTensor, RefTensor


class pimLinearFunction(Function):
    @staticmethod
    def forward(ctx, qinput: NormalTensor, inputArr: ArrayTensor, weight: ArrayTensor, weight_t: ArrayTensor, hasBias: bool):
        if hasBias:
            qinput.add_additional_one()

        inputArr.write_normal(qinput)
        qoutput = qinput.matmul_array(weight)

        ctx.inputArr = inputArr
        ctx.weight_t = weight_t
        ctx.hasBias = hasBias 

        return qoutput 

    @staticmethod
    def backward(ctx, qgrad_output):
        inputArr = ctx.inputArr
        weight_t = ctx.weight_t
        hasBias = ctx.hasBias 
        
        qgrad_input = qgrad_output.mul(weight_t)
        if hasBias:
            qgrad_input.remove_additaional_one()
        delta_weight_t = qgrad_output.t_matmul_array(inputArr)

        return qgrad_input, None, None, delta_weight_t, None 

# optimi   weight.subt(weight_t.grad)
#         weight_t.sub(weight_t.grad)


class PIMLinear(torch.nn.Module):
    def __init__(self, m: int, n: int, batch_size: int, bitwidth: int, hasBias: bool, arrayMode: str, quantizerMode: str, absMaxValue: float):
        super().__init__()
        if hasBias:
            m += 1
        self.m, self.n, self.hasBias, self.maxValue = m, n, hasBias, absMaxValue
        # quantizer mode dynamic, static.
        self.quantizerMode = quantizerMode 

        if arrayMode == "RefTensor":
            self.wArr = RefTensor(bitwidth)
            self.wtArr = RefTensor(bitwidth)
            self.inputArr = RefTensor(bitwidth)
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
        qoutput = pimLinearFunction.apply(qinput, self.inputArr, self.wArr, self.wtArr, self.hasBias)
        return qoutput


class deQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: NormalTensor, x: NormalTensor, bit: int):
        ctx.x = x
        ctx.bit = bit
        return qinput.de_quantization()

    @staticmethod
    def backward(ctx, grad_output):        
        x = ctx.x
        x.quantization(grad_output, grad_output.abs().max(), ctx.bit)
        return x.data


class deQuanLayer(torch.nn.Module):
    def __init__(self, bitwidth: int, quantizerMode: int):
        super().__init__()
        self.x = NormalTensor()
        self.bit = bitwidth
        self.quantizerMode = quantizerMode

    def forward(self, qinput: NormalTensor):
        return deQuanFunction.apply(qinput, self.x, self.bit)


if __name__ == "__main__":
    pass
