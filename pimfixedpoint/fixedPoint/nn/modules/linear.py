import torch
from .. import functional as fpF
from torch.nn import Module, Parameter, init
import math
from .. import fixedPointArithmetic as fpA
from ..commonConst import TensorType, data_flow_bit_width, torch_float


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 output_bit_width: int = data_flow_bit_width, weight_bit_width: int = data_flow_bit_width,
                 grad_output_bit_width: int = data_flow_bit_width, compute_weight_bit_width: int = None,
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
        temp_weight = torch.empty(self.out_features, self.in_features, dtype=torch_float)
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
            = fpF.linear.apply(qinput, qinput_config, self.fp_weight, self.fp_weight_cfg, self.hasBias,
                               self.outputBits, self.gradOutputBits, self.computeWeightBits, self.training)

        return qoutput, qoutput_config

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.hasBias
        )


if __name__ == "__main__":
    pass
