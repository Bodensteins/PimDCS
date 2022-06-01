import torch
from .. import functional as fpF
from torch.nn import Module, Parameter, init
import math
from ..fixedPointArithmetic import creat_quantization_para, quantization_tensor
from ..commonConst import TensorType, half_data_flow_bit_width


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None,
                 input_bit_width: int = half_data_flow_bit_width, weight_bit_width: int = half_data_flow_bit_width,
                 grad_output_bit_width: int = half_data_flow_bit_width, static_tensor_mode: str = "NormalTensor",
                 quantizer_mode: str = ""):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hasBias = bias
        self.device = device

        # quantizer mode dynamic, static.
        self.quantizerMode = quantizer_mode
        self.inputBits = input_bit_width
        self.gradOutputBits = grad_output_bit_width
        self.fp_weight = None
        self.pim_weight = None  # this is float weight will be  delete

        if static_tensor_mode == "NormalTensor":
            # self.pim_inArr_cfg = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal,
            #                                              device=device)
            self.fp_weight_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weight_bit_width, tensor_type=TensorType.Normal, device=device))
            # self.pim_wtArr_cfg = torch.nn.Parameter(
            #     creat_quantization_para(bit_width=weight_bit_width, tensor_type=TensorType.Normal, device=device))

            # self.pim_inArr = torch.empty([batch_size, in_features], dtype=torch_float, device=device)
        else:
            raise Exception("We don't implement this tensor mode: " + static_tensor_mode)

        # this bit_width is not used in fact.
        self.fp_delta_weight_cfg = Parameter(
            creat_quantization_para(bit_width=weight_bit_width, tensor_type=TensorType.Normal, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty(self.out_features, self.in_features, device=self.device)
        init.kaiming_uniform_(temp_weight, math.sqrt(5))
        # temp_weight = temp_weight.T

        if self.hasBias:
            temp_bias = torch.empty(self.out_features, device=self.device)
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(1)
            temp_weight = torch.cat((temp_weight, temp_bias), 1)

        self.pim_weight = torch.nn.Parameter(temp_weight.clone().detach().requires_grad_())
        self.fp_weight = torch.nn.Parameter(
            quantization_tensor(self.fp_weight_cfg, temp_weight))
        # self.pim_wtArr = torch.nn.Parameter(
        #     quantization_tensor(self.pim_wtArr_cfg, temp_weight.t()))

    def forward(self, qinput, qinput_config, _input=None):
        qoutput, qoutput_config, _ = fpF.linear.apply(qinput, qinput_config, self.fp_weight, self.fp_weight_cfg,
                                                      self.fp_delta_weight_cfg, self.inputBits,
                                                      self.gradOutputBits, self.hasBias, _input, self.pim_weight)

        return qoutput, qoutput_config

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.hasBias
        )


if __name__ == "__main__":
    pass
