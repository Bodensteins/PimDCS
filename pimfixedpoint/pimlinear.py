from os import stat
import torch
import math
from torch.autograd import Function, grad
from torch import Tensor
from quantization import add_additional_col_of_one, change_bit_width_, parse_float_tensor_list, write_array_, \
    normal_matmul_array, \
    remove_additional_col, normal_t_matmul_array, TensorType, creat_quantization_para, quantization_tensor, \
    parse_quantization_para, de_quantization, quantization_tensor_less, float_to_int, add_additional_col_of_zero


# torch.set_printoptions(profile="full")

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
                hasBias: bool,
                input: Tensor,
                weight: Tensor):

        if hasBias:
            qinput = add_additional_col_of_one([qinput, qinput_config])
            if input != None:
                full_one_col = torch.full([input.size()[0], 1], 1, dtype=input.dtype, device=input.device)
                input = torch.cat((input, full_one_col), 1)

        ctx.has_origin_input = False
        output = None
        if input != None:
            output = torch.matmul(input, weight)
            ctx.input = input
            ctx.weight = weight
            ctx.has_origin_input = True

        change_bit_width_([qinput, qinput_config], inputBits)

        if qinputArr is None:
            qinputArr = write_array_([qinputArr, qinputArr_config], [qinput, qinput_config])
        else:
            write_array_([qinputArr, qinputArr_config], [qinput, qinput_config])

        qoutput, qoutput_config = normal_matmul_array([qinput, qinput_config], [qweight, qweight_config])

        ctx.save_for_backward(qinputArr, qweight_t)
        ctx.qinputArr_config = qinputArr_config
        ctx.qweight_t_config = qweight_t_config
        ctx.delta_qweight_t_config = delta_qweight_t_config

        ctx.gradOutputBits = gradOutputBits
        ctx.hasBias = hasBias

        return qoutput, qoutput_config, output

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config, grad_output):
        qinputArr, qweight_t = ctx.saved_tensors
        qinputArr_config = ctx.qinputArr_config
        qweight_t_config = ctx.qweight_t_config
        delta_qweight_t_config = ctx.delta_qweight_t_config
        # print(grad_output)
        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits
        change_bit_width_([qgrad_output, qgrad_output_config], qgrad_output_bits)

        qgrad_input, qgrad_input_config = normal_matmul_array([qgrad_output, qgrad_output_config],
                                                              [qweight_t, qweight_t_config])

        grad_input = None
        d_w = None
        if ctx.has_origin_input:
            grad_input = torch.matmul(grad_output, ctx.weight.t())
            d_w = torch.matmul(grad_output.t(), ctx.input).t()

        if hasBias:
            qgrad_input = remove_additional_col(qgrad_input)
            if ctx.has_origin_input:
                grad_input = grad_input[:, 0:-1]

        # print(f'qgradout= {de_quantization([qgrad_output, qgrad_output_config])}')
        # print(f'qintputarr= {de_quantization([qinputArr, qinputArr_config])}')
        delta_qweight_t, _ = normal_t_matmul_array(
            [qgrad_output, qgrad_output_config], [qinputArr, qinputArr_config], delta_qweight_t_config)

        # print(f'delta= {de_quantization([delta_qweight_t, delta_qweight_t_config])}')
        delta_qweight_t = add_additional_col_of_zero([delta_qweight_t, delta_qweight_t_config])

        # print(float_to_int(delta_qweight_t))
        # print('----------------')
        # print(f"delta_weight_t = {delta_qweight_t.data_ptr()}")
        return qgrad_input, qgrad_input_config, None, None, None, None, delta_qweight_t, None, None, None, None, None, grad_input, d_w


class PimLinear(torch.nn.Module):
    def __init__(self, m: int, n: int, inputBits: int = 16, weightBits: int = 16, gradOutputBits: int = 16,
                 arrayMode: str = "RefTensor", quantizerMode: str = "", absMaxValueLeft = None, absMaxValueRight = None, hasBias: bool = True,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        if hasBias:
            m += 1
        self.m, self.n, self.hasBias, self.maxValueLeft, self.maxValueRight = m, n, hasBias, absMaxValueLeft, absMaxValueRight
        # quantizer mode dynamic, static.
        self.quantizerMode = quantizerMode 
        self.device = device
        self.inputBits, self.gradOutputBits = inputBits, gradOutputBits
        self.wArr, self.wtArr, self.inputArr = None, None, None

        # this bit_width is not used in fact.
        self.delta_qweight_t_config = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal,
                                                              device=device)

        if arrayMode == "RefTensor":
            self.inputArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref,
                                                          device=device)
            self.wArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref,
                                                      device=device)
            self.wtArrConfig = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref,
                                                       device=device)
        # elif arrayMode == "PNTensor":
        #    self.wArr = PNTensor(m, n, absMaxValue, bitwidth)
        #    self.wtArr = PNTensor(n, m, absMaxValue, bitwidth) 
        #    self.inputArr = PNTensor(batch_size, m, absMaxValue, bitwidth)
        else:
            raise Exception("We don't implement this array mode!", arrayMode)
        
        self.weight_init()
    
    def weight_init(self):
        temp_weight = torch.empty(self.m, self.n, device=self.device)
        torch.nn.init.kaiming_uniform_(temp_weight, math.sqrt(5))

        if self.hasBias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight[0:-1])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_weight[-1], -bound, bound)

        self.weight = torch.nn.Parameter(temp_weight.clone().detach().requires_grad_())
        #self.delta_qweight_t_config = torch.nn.Parameter(self.delta_qweight_t_config)
        self.wArr = torch.nn.Parameter(quantization_tensor(self.wArrConfig, temp_weight, self.maxValueLeft, self.maxValueRight))
        self.wtArr = torch.nn.Parameter(quantization_tensor(self.wtArrConfig, temp_weight.t(), self.maxValueLeft, self.maxValueRight))

    def forward(self, qinput: Tensor, qinput_config: Tensor, input: Tensor = None):
        qoutput, qoutput_config, _ = PimLinearFunction.apply(qinput, qinput_config, self.inputArr,
                                                               self.inputArrConfig, self.wArr, self.wArrConfig,
                                                               self.wtArr,
                                                               self.wtArrConfig, self.delta_qweight_t_config,
                                                               self.inputBits, self.gradOutputBits, self.hasBias, input,
                                                               self.weight)

        return qoutput, qoutput_config


class DeQuanFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, bit: int, backBit: int, quantizerMode: str = "dynamic"):
        if quantizerMode == "dynamic":
            _, ctx.backBit, _ = parse_quantization_para(float_to_int(qinput_config))
        else:
            ctx.backBit = backBit
        
        if bit is not None:
            change_bit_width_([qinput, qinput_config], bit)

        return de_quantization([qinput, qinput_config])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        qgrad_output_config = creat_quantization_para(bit_width=ctx.backBit, tensor_type=TensorType.Normal,
                                                      device=grad_output.device)

        qgrad_output = quantization_tensor(qgrad_output_config, grad_output)

        return qgrad_output, qgrad_output_config, None, None, None


class DeQuanLayer(torch.nn.Module):
    def __init__(self, bitWidth: int = 16, backBitWidth: int = 16, quantizerMode: str = "dynamic"):
        super().__init__()
        self.quantizerMode = quantizerMode
        self.bit = bitWidth
        self.backBit = backBitWidth

    def forward(self, qinput: Tensor, qinput_config: Tensor):
        return DeQuanFunction.apply(qinput, qinput_config, self.bit, self.backBit, self.quantizerMode)


class quanFunction(Function):
    @staticmethod
    def forward(ctx, input, bit):
        qoutput_config = creat_quantization_para(bit_width=bit, tensor_type=TensorType.Normal, device=input.device)
        qoutput = quantization_tensor(qoutput_config, input)
        return qoutput, qoutput_config

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        return de_quantization([qgrad_output, qgrad_output_config]), None


class ReluFunction(Function):
    @staticmethod
    def forward(ctx, qinput: Tensor, qinput_config: Tensor, origin_input: Tensor):
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


class PimRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput: Tensor, qinput_config: Tensor, origin_input: Tensor = None):
        qoutput, qoutput_config, _ = ReluFunction.apply(qinput, qinput_config, origin_input)
        return qoutput, qoutput_config


if __name__ == "__main__":
    pass
