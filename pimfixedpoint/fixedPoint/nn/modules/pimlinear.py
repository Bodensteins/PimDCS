import torch
import math
from torch.autograd import Function
from torch import Tensor
from fixedPoint.fixedPointArithmetic import add_additional_col_of_one, set_bit_width_, write_array_, quantization_matmul, \
    remove_additional_col, quantization_t_matmul, TensorType, creat_quantization_para, quantization_tensor, \
    to_int, add_additional_col_of_zero, \
    pow_2_n, torch_float


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
            if input is not None:
                full_one_col = torch.full([input.size()[0], 1], 1, dtype=input.dtype, device=input.device)
                input = torch.cat((input, full_one_col), 1)

        ctx.has_origin_input = False
        output = None
        if input is not None:
            output = torch.matmul(input, weight)
            ctx.input = input
            ctx.pim_weight = weight
            ctx.has_origin_input = True

        if qinputArr is not None:
            write_array_([qinputArr, qinputArr_config], [qinput, qinput_config])

        set_bit_width_([qinput, qinput_config], inputBits)

        qoutput, qoutput_config = quantization_matmul([qinput, qinput_config], [qweight, qweight_config])

        ctx.save_for_backward(qinputArr, qweight_t)
        ctx.qinputArr_config = qinputArr_config
        ctx.qweight_t_config = qweight_t_config
        ctx.pim_delta_qweight_t_cfg = delta_qweight_t_config

        ctx.gradOutputBits = gradOutputBits
        ctx.hasBias = hasBias

        return qoutput, qoutput_config, output

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config, grad_output):
        qinputArr, qweight_t = ctx.saved_tensors
        qinputArr_config = ctx.qinputArr_config
        qweight_t_config = ctx.qweight_t_config
        delta_qweight_t_config = ctx.pim_delta_qweight_t_cfg
        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits
        set_bit_width_([qgrad_output, qgrad_output_config], qgrad_output_bits)

        qgrad_input, qgrad_input_config = quantization_matmul([qgrad_output, qgrad_output_config],
                                                              [qweight_t, qweight_t_config])

        grad_input = None
        d_w = None
        if ctx.has_origin_input:
            grad_input = torch.matmul(grad_output, ctx.pim_weight.t())
            d_w = torch.matmul(grad_output.t(), ctx.input).t()

        if hasBias:
            qgrad_input = remove_additional_col(qgrad_input)
            if ctx.has_origin_input:
                grad_input = grad_input[:, 0:-1]

        delta_qweight_t, _ = quantization_t_matmul([qgrad_output, qgrad_output_config], [qinputArr, qinputArr_config],
                                                   delta_qweight_t_config)
        # print(f'err shape: {qgrad_output.size()} err cfg: {to_int(qgrad_output_config)} '
        #       f'input shape: {qinputArr.size()} input cfg: {to_int(qinputArr_config)} '
        #       f'grad shape: {delta_qweight_t.size()} grad cfg: {to_int(delta_qweight_t_config)}')

        # print(f'delta= {de_quantization([delta_qweight_t, delta_qweight_t_config])}')
        delta_qweight_t = add_additional_col_of_zero([delta_qweight_t, delta_qweight_t_config])

        return qgrad_input, qgrad_input_config, None, None, None, None, delta_qweight_t, None, None, None, None, None, \
               grad_input, d_w


class PimLinear(torch.nn.Module):
    def __init__(self, m: int, n: int, batch_size: int, inputBits: int = 16, weightBits: int = 16,
                 gradOutputBits: int = 16, arrayMode: str = "RefTensor", quantizerMode: str = "",
                 absMaxValueLeft=None, absMaxValueRight=None, hasBias: bool = True,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        if hasBias:
            m += 1
        self.m, self.n, self.hasBias, self.maxValueLeft, self.maxValueRight = m, n, hasBias, absMaxValueLeft, \
                                                                              absMaxValueRight
        # quantizer mode dynamic, static.
        self.quantizerMode = quantizerMode
        self.device = device
        self.inputBits, self.gradOutputBits = inputBits, gradOutputBits
        self.pim_wArr, self.pim_wtArr, self.pim_inArr = None, None, None

        # this bit_width is not used in fact.
        self.pim_delta_qweight_t_cfg = torch.nn.Parameter(
            creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))

        if arrayMode == "RefTensor":
            self.pim_inArr_cfg = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref,
                                                         device=device)
            self.pim_wArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref, device=device))
            self.pim_wtArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Ref, device=device))

            neg_levels = pow_2_n(weightBits - 1)
            self.pim_inArr = torch.empty([batch_size, m + 1], dtype=torch_float, device=device)
            int_input_array = to_int(self.pim_inArr)
            int_input_array[:, -1] = neg_levels
        else:
            raise Exception("We don't implement this array mode!", arrayMode)

        self.weight_init()

    def weight_init(self):
        temp_weight = torch.empty(self.n, self.m - 1, device=self.device)
        torch.nn.init.kaiming_uniform_(temp_weight, math.sqrt(5))
        temp_weight = temp_weight.T

        if self.hasBias:
            temp_bias = torch.empty(self.n, device=self.device)
            _, fan_in = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight)  # our weight is different
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(0)
            temp_weight = torch.cat((temp_weight, temp_bias), 0)

        self.pim_weight = torch.nn.Parameter(temp_weight.clone().detach().requires_grad_())
        self.pim_wArr = torch.nn.Parameter(
            quantization_tensor(self.pim_wArr_cfg, temp_weight, self.maxValueLeft, self.maxValueRight))
        self.pim_wtArr = torch.nn.Parameter(
            quantization_tensor(self.pim_wtArr_cfg, temp_weight.t(), self.maxValueLeft, self.maxValueRight))

    def forward(self, qinput: Tensor, qinput_config: Tensor, input: Tensor = None):
        if self.training:
            qoutput, qoutput_config, _ = PimLinearFunction.apply(qinput, qinput_config, self.pim_inArr,
                                                                 self.pim_inArr_cfg, self.pim_wArr, self.pim_wArr_cfg,
                                                                 self.pim_wtArr, self.pim_wtArr_cfg,
                                                                 self.pim_delta_qweight_t_cfg, self.inputBits,
                                                                 self.gradOutputBits, self.hasBias, input, self.pim_weight)
        else:
            qoutput, qoutput_config, _ = PimLinearFunction.apply(qinput, qinput_config, None, self.pim_inArr_cfg,
                                                                 self.pim_wArr, self.pim_wArr_cfg, self.pim_wtArr,
                                                                 self.pim_wtArr_cfg, self.pim_delta_qweight_t_cfg,
                                                                 self.inputBits, self.gradOutputBits, self.hasBias,
                                                                 input, self.pim_weight)

        return qoutput, qoutput_config


if __name__ == "__main__":
    pass
