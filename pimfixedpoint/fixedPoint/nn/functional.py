from torch.autograd import Function
from . import fixedPointArithmetic as fpA
from .commonConst import TensorType, torch_int
import torch
import pydevd

debug_backward = False


class dequan(Function):
    @staticmethod
    def forward(ctx, qinput, qinput_config, bit: int, backBit: int, quantizerMode: str = "dynamic"):
        if quantizerMode == "dynamic":
            _, ctx.backBit, _ = fpA.parse_quantization_para(fpA.to_int(qinput_config))
        else:
            ctx.backBit = backBit

        if bit is not None:
            fpA.set_bit_width_([qinput, qinput_config], bit)

        return fpA.de_quantization([qinput, qinput_config])

    @staticmethod
    def backward(ctx, grad_output):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        qgrad_output_config = fpA.creat_quantization_para(bit_width=ctx.backBit, tensor_type=TensorType.Normal,
                                                      device=grad_output.device)
        # print(grad_output)
        qgrad_output = fpA.quantization_tensor(qgrad_output_config, grad_output)

        return qgrad_output, qgrad_output_config, None, None, None


class quan(Function):
    @staticmethod
    def forward(ctx, _input, bit_width):
        fp_output_config = fpA.creat_quantization_para(bit_width=bit_width, tensor_type=TensorType.Normal,
                                                   device=_input.device)
        fp_output = fpA.quantization_tensor(fp_output_config, _input)
        return fp_output, fp_output_config

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_config):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        return fpA.de_quantization([fp_grad_output, fp_grad_output_config]), None


class relu(Function):
    @staticmethod
    def forward(ctx, qinput, qinput_config):
        neg_position = fpA.fixed_point_less([qinput, qinput_config], 0)
        qinput[neg_position] = 0  # The zero in ieee754 is all zero as well.
        ctx.save_for_backward(neg_position)
        return qinput, qinput_config

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        neg_position, = ctx.saved_tensors
        qgrad_output[neg_position] = 0

        return qgrad_output, qgrad_output_config


class dropout(Function):
    @staticmethod
    def forward(ctx, fake_input, fake_input_cfg, dropout_ratio, training):
        if training:
            mask = torch.rand_like(fake_input) < dropout_ratio  # position of smaller than dropout_ratio
            ctx.save_for_backward(mask)  # tensor should be saved in save for backward
            fake_input[mask] = 0  # The zero in ieee754 is all zero as well.
            return fake_input, fake_input_cfg
        else:
            int_input, _ = fpA.parse_tensor_list_to_int([fake_input, fake_input_cfg])
            int_input = int_input.mul(1 - dropout_ratio).to(dtype=torch_int)

            return fpA.to_float(int_input), fake_input_cfg

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_cfg):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        mask, = ctx.saved_tensors
        fp_grad_output[mask] = 0
        return fp_grad_output, fp_grad_output_cfg, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_config, fp_weight, fp_weight_config, delta_fp_weight_config,
                input_bit_width: int, grad_output_bit_width: int, has_bias: bool, _input, weight):
        if has_bias:
            fp_input = fpA.add_additional_col_of_one([fp_input, fp_input_config])
            if _input is not None:
                full_one_col = torch.full([_input.size()[0], 1], 1, dtype=_input.dtype, device=_input.device)
                _input = torch.cat((_input, full_one_col), 1)

        ctx.has_origin_input = False
        output = None
        if _input is not None:
            # todo: modify here, may have bugs
            output = torch.matmul(_input, weight)
            ctx.input = _input
            ctx.pim_weight = weight
            ctx.has_origin_input = True

        # todo: optimize here
        ctx.save_for_backward(fp_input.clone(), fp_input_config.clone(), fp_weight, fp_weight_config,
                              delta_fp_weight_config)

        fpA.set_bit_width_([fp_input, fp_input_config], input_bit_width)

        qoutput, qoutput_config = fpA.fixed_point_matmul([fp_input, fp_input_config], [fp_weight.t(), fp_weight_config])

        ctx.gradOutputBits = grad_output_bit_width
        ctx.hasBias = has_bias

        return qoutput, qoutput_config, output

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config, grad_output):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        qinputArr, qinputArr_config, qweight, qweight_config, delta_qweight_config = ctx.saved_tensors
        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits
        fpA.set_bit_width_([qgrad_output, qgrad_output_config], qgrad_output_bits)

        qgrad_input, qgrad_input_config = fpA.fixed_point_matmul([qgrad_output, qgrad_output_config],
                                                                 [qweight, qweight_config])

        grad_input = None
        d_w = None
        if ctx.has_origin_input:
            grad_input = torch.matmul(grad_output, ctx.pim_weight.t())
            d_w = torch.matmul(grad_output.t(), ctx.input).t()

        if hasBias:
            qgrad_input = qgrad_input[:, 0:-1]
            if ctx.has_origin_input:
                grad_input = grad_input[:, 0:-1]

        # todo: may be simplified
        delta_qweight, _ = \
            fpA.fixed_point_t_matmul([qgrad_output, qgrad_output_config], [qinputArr, qinputArr_config],
                                     delta_qweight_config)

        return qgrad_input, qgrad_input_config, delta_qweight, None, None, None, None, None, grad_input, d_w
