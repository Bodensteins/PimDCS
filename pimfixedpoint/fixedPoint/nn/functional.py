from torch.autograd import Function
from . import fixedPointArithmetic as fpA
from .commonConst import TensorType, torch_int, data_flow_bit_width
import torch
import pydevd

debug_backward = False

if debug_backward:
    print("debug mod, can't run")


class dequan(Function):
    @staticmethod
    def forward(ctx, qinput, qinput_config, bit: int, backBit: int, quantizerMode: str = "dynamic"):
        if quantizerMode == "dynamic":
            ctx.backBit = data_flow_bit_width
            # _, ctx.backBit, _ = fpA.parse_quantization_para(fpA.to_int(qinput_config))
        else:
            ctx.backBit = backBit

        if bit is not None:
            fpA.set_bit_width_((qinput, qinput_config), bit)

        return fpA.de_quantization((qinput, qinput_config))

    @staticmethod
    def backward(ctx, grad_output):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        qgrad_output, qgrad_output_config = fpA.quantization_tensor(grad_output, ctx.backBit, TensorType.Normal)

        return fpA.to_float(qgrad_output), fpA.to_float(qgrad_output_config), None, None, None


class quan(Function):
    @staticmethod
    def forward(ctx, _input, bit_width):
        fp_output, fp_output_config = fpA.quantization_tensor(_input, bit_width, TensorType.Normal)
        return fpA.to_float(fp_output), fpA.to_float(fp_output_config)

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_config):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        return fpA.de_quantization((fp_grad_output, fp_grad_output_config)), None


class relu(Function):
    @staticmethod
    def forward(ctx, qinput, qinput_config):
        neg_position = fpA.fixed_point_less((qinput, qinput_config), 0)
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
    def forward(ctx, fp_input, fp_input_cfg, dropout_ratio, training):
        if training:
            mask = torch.rand_like(fp_input) < dropout_ratio  # position of smaller than dropout_ratio
            ctx.save_for_backward(mask)  # tensor should be saved in save for backward
            fp_input[mask] = 0  # The zero in ieee754 is all zero as well.
            return fp_input, fp_input_cfg
        else:
            int_input = fpA.to_int(fp_input)
            int_input = int_input.mul(1 - dropout_ratio).to(dtype=torch_int)

            return fpA.to_float(int_input), fp_input_cfg

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_cfg):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        mask, = ctx.saved_tensors
        fp_grad_output[mask] = 0
        return fp_grad_output, fp_grad_output_cfg, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_config, fp_weight, fp_weight_config, has_bias: bool,
                input_bit_width: int, grad_output_bit_width: int):
        if has_bias:
            fp_input = fpA.add_additional_col_of_one((fp_input, fp_input_config))

        # todo: optimize here
        ctx.save_for_backward(fp_input, fp_input_config.clone(), fp_weight, fp_weight_config)
        # ctx.save_for_backward(fp_input, fp_input_config, fp_weight, fp_weight_config)

        # fpA.set_bit_width_((fp_input, fp_input_config), input_bit_width)

        # clone_weight, clone_config = fpA.get_clone((fp_weight, fp_weight_config), 8)
        # qoutput, qoutput_config = fpA.fixed_point_matmul((fp_input, fp_input_config), (clone_weight.t(), clone_config))

        qoutput, qoutput_config \
            = fpA.fixed_point_matmul((fp_input, fp_input_config), (fp_weight.t(), fp_weight_config), input_bit_width)

        ctx.gradOutputBits = grad_output_bit_width
        ctx.hasBias = has_bias

        return fpA.to_float(qoutput), fpA.to_float(qoutput_config)

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        if debug_backward is True:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        qinputArr, qinputArr_config, qweight, qweight_config = ctx.saved_tensors
        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits
        # fpA.set_bit_width_((qgrad_output, qgrad_output_config), qgrad_output_bits)

        # clone_weight, clone_config = fpA.get_clone((qweight, qweight_config), 8)
        # qgrad_input, qgrad_input_config = fpA.fixed_point_matmul((qgrad_output, qgrad_output_config),
        #                                                          (clone_weight, clone_config))
        qgrad_input, qgrad_input_config \
            = fpA.fixed_point_matmul((qgrad_output, qgrad_output_config), (qweight, qweight_config), qgrad_output_bits)
        # print(f'backward: {qgrad_output.shape}')

        if hasBias:
            qgrad_input = qgrad_input[:, 0:-1]

        fp_delta_weight, fp_delta_weight_cfg \
            = fpA.fixed_point_matmul((qgrad_output.t(), qgrad_output_config), (qinputArr, qinputArr_config))

        fp_grad_input = fpA.to_float(qgrad_input)
        fp_grad_input_cfg = fpA.to_float(qgrad_input_config)
        fp_grad_weight = fpA.to_float(fp_delta_weight)
        fp_grad_weight_cfg = fpA.to_float(fp_delta_weight_cfg)

        return fp_grad_input, fp_grad_input_cfg, fp_grad_weight, fp_grad_weight_cfg, None, None, None
