from os import stat
import torch
import math
from torch.autograd import Function, grad
from torch import Tensor
from quantization import add_additional_col_of_one, change_bit_width_, parse_float_tensor_list, write_array, normal_matmul_array, \
    remove_additional_col, normal_t_matmul_array, TensorType, creat_quantization_para, quantization_tensor, \
    parse_quantization_para, de_quantization, quantization_tensor_less, float_to_int, add_additional_col_of_zero

class PimLinearFunction(Function):
    @staticmethod
    def forward(ctx,
                qinput: Tensor,
                qinput_config: Tensor,
                qinputArr: Tensor,
                qinputArr_config: Tensor,
                qweight: Tensor,
                qweight_config: Tensor,
                qweight_t: Tensor,
                qweight_t_config: Tensor,
                delta_qweight_t_config: Tensor,
                inputBits: int,
                gradOutputBits: int,
                hasBias: bool):

        if hasBias:
            qinput = add_additional_col_of_one([qinput, qinput_config])
        
        change_bit_width_([qinput, qinput_config], inputBits)
        qinputArr = write_array([qinputArr, qinputArr_config], [qinput, qinput_config])
        qoutput, qoutput_config = normal_matmul_array([qinput, qinput_config], [qweight, qweight_config])

        ctx.save_for_backward(qinputArr, qweight_t)
        ctx.qinputArr_config = qinputArr_config
        ctx.delta_qweight_t_config = delta_qweight_t_config

        ctx.gradOutputBits = gradOutputBits
        ctx.hasBias = hasBias

        return qoutput, qoutput_config

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        qinputArr, qweight_t = ctx.saved_tensors
        qinputArr_config = ctx.qinputArr_config
        qweight_t_config = ctx.qweight_t_config
        delta_qweight_t_config = ctx.delta_qweight_t_config

        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits

        change_bit_width_([qgrad_output, qgrad_output_config], qgrad_output_bits)
        
        qgrad_input, qgrad_input_config = normal_matmul_array([qgrad_output, qgrad_output_config], [qweight_t, qweight_t_config])

        if hasBias:
            qgrad_input = remove_additional_col(qgrad_input)


        delta_qweight_t, delta_qweight_t_config = normal_t_matmul_array([qgrad_output, qgrad_output_config], [qinputArr, qinputArr_config])
        delta_qweight_t = add_additional_col_of_zero([delta_qweight_t, delta_qweight_t_config])

        return qgrad_input, qgrad_input_config, None, None, None, None, delta_qweight_t, None, None, None, None, None


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
        self.delta_qweight_t_config = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal) 

        if arrayMode == "RefTensor":
            self.inputArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref)
            self.wArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref)
            self.wtArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref)
        # elif arrayMode == "PNTensor":
        #    self.wArr = PNTensor(m, n, absMaxValue, bitwidth)
        #    self.wtArr = PNTensor(n, m, absMaxValue, bitwidth) 
        #    self.inputArr = PNTensor(batch_size, m, absMaxValue, bitwidth)
        else:
            raise Exception("We don't implement this array mode!", arrayMode)

        self.weight_init()
    
    def weight_init(self):
        temp_weight = torch.empty(self.m, self.n)
        torch.nn.init.kaiming_uniform_(temp_weight, math.sqrt(5))

        if self.hasBias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight[0:-1])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_weight[-1], -bound, bound)

        self.wArrConfig = torch.nn.Parameter(self.wArrConfig)
        self.wtArrConfig = torch.nn.Parameter(self.wtArrConfig)
        self.wArr = torch.nn.Parameter(quantization_tensor(self.wArrConfig, temp_weight, max(self.maxValue, temp_weight.abs().max())))
        self.wtArr = torch.nn.Parameter(quantization_tensor(self.wtArrConfig, temp_weight.t(), max(self.maxValue, temp_weight.abs().max())))


    def forward(self, qinput: Tensor, qinput_config):
        qoutput, qoutput_config = PimLinearFunction.apply(qinput, qinput_config, self.inputArr, self.inputArrConfig, self.wArr, self.wArrConfig, self.wtArr, 
            self.wtArrConfig, self.delta_qweight_t_config, self.inputBits, self.gradOutputBits, self.hasBias)

        return qoutput, qoutput_config


class DeQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, bit: int, quantizerMode: str):
        if quantizerMode == "dynamic":
            _, ctx.bit, _ = parse_quantization_para(float_to_int(qinput_config))
        else:
            ctx.bit = bit

        return de_quantization([qinput, qinput_config])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        qgrad_output_config = creat_quantization_para(bit_width=ctx.bit, tensor_type=TensorType.Normal)
        qgrad_output = quantization_tensor(qgrad_output_config, grad_output, grad_output.abs().max())

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
        neg_position = quantization_tensor_less([qinput, qinput_config], 0)
        ctx.neg_position = neg_position
        qinput[neg_position] = 0  # The zero in ieee754 is all zero as well.
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
