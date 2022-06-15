import torch
import math
from torch import Tensor
import torch.nn.functional
from torch.nn import init
from .. import functional as fpF
from .. import fixedPointArithmetic as fpA
from ..fixedPointArithmetic import creat_quantization_para, quantization_tensor
from torch.nn import Module
from ..commonConst import TensorType
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', inputBits: int = 16, weightBits: int = 16, gradOutputBits: int = 16,
                 weight_tensor_mode: str = "NormalTensor", quantizerMode: str = ""):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.hasBias = bias
        self.padding_mode = padding_mode

        self.inputBits, self.weightBits, self.gradOutputBits = inputBits, weightBits, gradOutputBits
        self.quantizerMode = quantizerMode

        self.fp_weight = None

        if weight_tensor_mode == "NormalTensor":
            self.fp_weight_cfg = torch.nn.Parameter(creat_quantization_para(bit_width=weightBits,
                                                                            tensor_type=TensorType.Normal))
        else:
            raise Exception("We don't implement this tensor mode!", weight_tensor_mode)

        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty([self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]])
        init.kaiming_uniform_(temp_weight, math.sqrt(5))

        temp_weight = temp_weight.reshape(self.out_channels, -1)

        if self.hasBias:
            temp_bias = torch.empty(self.out_channels)
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(temp_bias, -bound, bound)
            temp_bias = temp_bias.unsqueeze(1)
            temp_weight = torch.cat((temp_weight, temp_bias), 1)

        self.fp_weight = torch.nn.Parameter(quantization_tensor(self.fp_weight_cfg, temp_weight))

    def forward(self, _input: Tensor):
        # reshape qinput to the matrix-shape
        # Here, for simplicity, we use dequan & unfold & quan to simulate fixed-point unfold.
        batch_size = _input.size()[0]
        input_h = _input.size()[2]
        input_w = _input.size()[3]

        _input = torch.nn.functional.unfold(_input, self.kernel_size, dilation=self.dilation, padding=self.padding,
                                            stride=self.stride)
        _input = _input.transpose(1, 2).reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        fp_input, qinput_config = fpF.quan.apply(_input, self.inputBits)

        # re-use linear function to get the answer
        qoutput, qoutput_config = fpF.linear.apply(fp_input, qinput_config, self.fp_weight, self.fp_weight_cfg,
                                                   self.hasBias, self.inputBits, self.gradOutputBits)

        # reshape the output to the conv-shape
        # Here, for simplicity, we use dequan & fold & quan to simulate fixed-point fold
        output = fpF.dequan.apply(qoutput, qoutput_config, None, None)
        output = output.reshape(batch_size, -1, self.out_channels).transpose(1, 2)

        output_size = (math.floor((input_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                                  / self.stride[0] + 1),
                       math.floor((input_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                                  / self.stride[1] + 1))
        output = torch.nn.functional.fold(output, output_size, (1, 1))

        return output

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.hasBias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
