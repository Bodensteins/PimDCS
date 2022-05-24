import torch
from .. import functional as fpF
from torch.nn import Module
import math
from ..fixedPointArithmetic import creat_quantization_para, quantization_tensor
from ..commonConst import TensorType, torch_float


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, batch_size: int, inputBits: int = 16, weightBits: int = 16,
                 gradOutputBits: int = 16, static_tensor_mode: str = "NormalTensor", quantizerMode: str = "",
                 bias: bool = True, device: torch.device = torch.device("cpu")):
        super().__init__()
        if bias:
            in_features += 1
        self.m, self.n, self.hasBias = in_features, out_features, bias
        # quantizer mode dynamic, static.
        self.quantizerMode = quantizerMode
        self.device = device
        self.inputBits, self.gradOutputBits = inputBits, gradOutputBits
        self.pim_wArr, self.pim_wtArr, self.pim_inArr = None, None, None

        # this bit_width is not used in fact.
        self.pim_delta_qweight_t_cfg = torch.nn.Parameter(
            creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))

        if static_tensor_mode == "NormalTensor":
            self.pim_inArr_cfg = creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal,
                                                         device=device)
            self.pim_wArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))
            self.pim_wtArr_cfg = torch.nn.Parameter(
                creat_quantization_para(bit_width=weightBits, tensor_type=TensorType.Normal, device=device))

            self.pim_inArr = torch.empty([batch_size, in_features], dtype=torch_float, device=device)
        else:
            raise Exception("We don't implement this tensor mode: " + static_tensor_mode)

        self.reset_parameters()

    def reset_parameters(self):
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
            quantization_tensor(self.pim_wArr_cfg, temp_weight))
        self.pim_wtArr = torch.nn.Parameter(
            quantization_tensor(self.pim_wtArr_cfg, temp_weight.t()))

    def forward(self, qinput, qinput_config, _input=None):
        if self.training:
            qoutput, qoutput_config, _ = fpF.linear.apply(qinput, qinput_config, self.pim_inArr,
                                                          self.pim_inArr_cfg, self.pim_wArr, self.pim_wArr_cfg,
                                                          self.pim_wtArr, self.pim_wtArr_cfg,
                                                          self.pim_delta_qweight_t_cfg, self.inputBits,
                                                          self.gradOutputBits, self.hasBias, _input, self.pim_weight)
        else:
            qoutput, qoutput_config, _ = fpF.linear.apply(qinput, qinput_config, None, self.pim_inArr_cfg,
                                                          self.pim_wArr, self.pim_wArr_cfg, self.pim_wtArr,
                                                          self.pim_wtArr_cfg, self.pim_delta_qweight_t_cfg,
                                                          self.inputBits, self.gradOutputBits, self.hasBias,
                                                          _input, self.pim_weight)

        return qoutput, qoutput_config


if __name__ == "__main__":
    pass
