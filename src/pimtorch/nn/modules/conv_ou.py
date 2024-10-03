import math
import torch
from torch import Tensor
# from torch.autograd import Function
from torch.nn import Module, Parameter, init, functional
from src.pimtorch.config.globalCfg import globalCfg
from src.pimtorch.nn import matMulManager as mmm
from src.pimtorch.nn import fixedPointDataAnalyze as fpDA
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from typing import Union


# 仅支持前向传播，不考虑训练
# 已测试
class Conv2d_OU(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, has_bias: bool = True,
                 padding_mode: str = 'zeros', input_bit_width: int = globalCfg.dataFlowBitWidth,
                 output_bit_width: int = globalCfg.dataFlowBitWidth, weight_bit_width: int = globalCfg.dataFlowBitWidth,
                 grad_output_bit_width: int = globalCfg.dataFlowBitWidth,
                 next_grad_output_bit_width: int = globalCfg.dataFlowBitWidth,
                 batch_norm=True) -> None:
        super().__init__()

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('inChannels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('outChannels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.has_bias = has_bias
        self.padding_mode = padding_mode
        self.input_bit_width = input_bit_width  # notice: input bits should be equal to out bits of last layer
        self.output_bit_width = output_bit_width
        self.weight_bit_width = weight_bit_width
        self.grad_output_bits = grad_output_bit_width
        self.next_grad_output_bits = next_grad_output_bit_width
        self.batch_norm = batch_norm

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = Parameter(torch.empty((out_channels, in_channels // groups, *kernel_size), dtype=torch.float))
        if self.has_bias:
            self.bias = Parameter(torch.empty(out_channels, dtype=torch.float))
        else:
            self.bias = None
        self.mm_manager = None
        self.data_analyzer = None

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fanIn, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fanIn) if fanIn > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def create_mat_mul_manager(self) -> None:
        # print("Conv create mmm")
        weight_t = self.weight.reshape(self.out_channels, -1).t()
        self.mm_manager = mmm.FixedPointMatMulManager(weight_t, self.weight_bit_width, self.input_bit_width)
        # self.mm_manager = mmm.FixedPointMatMulManager_OU(weight_t, self.weight_bit_width, self.input_bit_width)
        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
            self.data_analyzer = fpDA.FpDataAnalyzer(self.mm_manager)
            self.mm_manager.set_data_analyzer(self.data_analyzer)
            self.data_analyzer.update_weight_sparsity()

    def forward(self, input:Tensor) -> Tensor:
        if self.training:
            # print("train Conv2d_OU")
            if self.padding_mode != 'zeros':
                return functional.conv2d(input=functional.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight=self.weight, bias=self.bias, stride=self.stride,
                                padding=_pair(0), dilation=self.dilation, groups=self.groups)
            return functional.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            # print("multiply with mm_manager in Conv2d_OU")

            batch_size = input.size()[0]
            input_h = input.size()[2]
            input_w = input.size()[3]

            # unfold input tensor
            input = torch.nn.functional.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            input = input.transpose(1, 2).reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[1])

            # output = self.mm_manager.mat_mul(input)
            output = self.mm_manager.fake_mat_mul(input)

            if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
                self.data_analyzer.update_all_input_statistic()

            if self.has_bias:
                output += self.bias

            # recover output tensor shape
            output = output.reshape(batch_size, -1, self.out_channels).transpose(1, 2)
            output_size = (math.floor((input_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                                      / self.stride[0] + 1),
                           math.floor((input_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                                      / self.stride[1] + 1))
            output = torch.nn.functional.fold(output, output_size, (1, 1))

            return output
        
    def print_statistic(self):
        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
            self.data_analyzer.print_statistic()

    def save_statistic(self):
        pass

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
        if not self.has_bias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)