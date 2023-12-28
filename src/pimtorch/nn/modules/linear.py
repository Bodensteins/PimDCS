import math
import torch
from torch.autograd import Function
from torch.nn import Module, Parameter, init
from src.pimtorch.nn import fixedPointArithmetic as fpA
from src.pimtorch.config.globalCfg import TensorType, globalCfg

class linear(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_config, fp_weight, fp_weight_config, has_bias: bool,
                output_bit_width: int, grad_output_bit_width: int, compute_weight_bit_width: int, training=True):
        if has_bias:
            fp_input, fp_input_config = fpA.add_additional_col_of_one((fp_input, fp_input_config))

        if compute_weight_bit_width is not None:
            fp_weight, fp_weight_config = fpA.get_clone((fp_weight, fp_weight_config), compute_weight_bit_width)

        if training:
            ctx.save_for_backward(fp_input, fp_input_config, fp_weight, fp_weight_config)
            ctx.gradOutputBits = grad_output_bit_width
            ctx.hasBias = has_bias

        qoutput, qoutput_config = fpA.fixed_point_matmul((fp_input, fp_input_config),
                                                         (fp_weight.t(), fp_weight_config), output_bit_width)

        return fpA.to_float(qoutput), fpA.to_float(qoutput_config)

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        # if debug_backward is True:
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)

        qinputArr, qinputArr_config, qweight, qweight_config = ctx.saved_tensors
        hasBias = ctx.hasBias
        qgrad_output_bits = ctx.gradOutputBits

        qgrad_input, qgrad_input_config \
            = fpA.fixed_point_matmul((qgrad_output, qgrad_output_config), (qweight, qweight_config), qgrad_output_bits)

        if hasBias:
            qgrad_input = qgrad_input[:, 0:-1]

        fp_delta_weight, fp_delta_weight_cfg \
            = fpA.fixed_point_matmul((qgrad_output.t(), qgrad_output_config), (qinputArr, qinputArr_config))

        fp_grad_input = fpA.to_float(qgrad_input)
        fp_grad_input_cfg = fpA.to_float(qgrad_input_config)
        fp_grad_weight = fpA.to_float(fp_delta_weight)
        fp_grad_weight_cfg = fpA.to_float(fp_delta_weight_cfg)

        return fp_grad_input, fp_grad_input_cfg, fp_grad_weight, fp_grad_weight_cfg, None, None, None, None, None

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 output_bit_width: int = globalCfg.dataFlowBitWidth, weight_bit_width: int = globalCfg.dataFlowBitWidth,
                 grad_output_bit_width: int = globalCfg.dataFlowBitWidth, compute_weight_bit_width: int = None,
                 weight_tensor_mode: str = "NormalTensor"):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hasBias = bias

        self.outputBits = output_bit_width
        self.weightBits = weight_bit_width
        self.gradOutputBits = grad_output_bit_width
        self.computeWeightBits = compute_weight_bit_width

        self.fp_weight = None
        self.fp_weight_cfg = None
        self.weight_tensor_mode = None

        if weight_tensor_mode == "NormalTensor":
            self.weight_tensor_mode = TensorType.Normal
        else:
            raise Exception("We don't implement this tensor mode: " + weight_tensor_mode)

        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty(self.out_features, self.in_features, dtype=globalCfg.torchFloat)
        init.kaiming_uniform_(temp_weight, a=math.sqrt(5))

        if self.hasBias:
            temp_bias = torch.empty(self.out_features)
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(1)
            temp_weight = torch.cat((temp_weight, temp_bias), 1)

        weight, weight_cfg = fpA.quantization_tensor(temp_weight, self.weightBits, self.weight_tensor_mode)
        self.fp_weight = Parameter(fpA.to_float(weight))
        self.fp_weight_cfg = Parameter(fpA.to_float(weight_cfg))

    def reset_parameters_from_float_parameters(self, weight, bias):
        if self.hasBias:
            bias = bias.unsqueeze(1)
            weight = torch.cat((weight, bias), 1)

        fp_weight, fp_weight_cfg = fpA.quantization_tensor(weight, self.weightBits, self.weight_tensor_mode)
        fpA.fixed_point_copy_((self.fp_weight, self.fp_weight_cfg), (fp_weight, fp_weight_cfg))

    def forward(self, qinput, qinput_config):
        qoutput, qoutput_config \
            = linear.apply(qinput, qinput_config, self.fp_weight, self.fp_weight_cfg, self.hasBias,
                               self.outputBits, self.gradOutputBits, self.computeWeightBits, self.training)

        return qoutput, qoutput_config

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.hasBias
        )
