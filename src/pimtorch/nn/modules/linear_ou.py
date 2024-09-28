import math
import torch
from torch import Tensor
# from torch.autograd import Function
from torch.nn import Module, Parameter, init, functional
from src.pimtorch.nn import fixedPointArithmetic as fpA
from src.pimtorch.config.globalCfg import TensorType, globalCfg
from src.pimtorch.nn import fixedPointDataAnalyze as fpDA
from src.pimtorch.nn import matMulManager as mmm


# 仅支持前向传播，不考虑训练
# 未测试
class LinearOU(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_bit_width: int = globalCfg.dataFlowBitWidth, 
                 output_bit_width: int = globalCfg.dataFlowBitWidth, weight_bit_width: int = globalCfg.dataFlowBitWidth,
                 grad_output_bit_width: int = globalCfg.dataFlowBitWidth, compute_weight_bit_width: int = None) -> None:
        super(LinearOU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.input_bit_width = input_bit_width
        self.output_bit_width = output_bit_width
        self.weight_bit_width = weight_bit_width
        self.grad_output_bit_width = grad_output_bit_width
        self.compute_weight_bit_width = compute_weight_bit_width

        self.weight = None
        self.mat_mul_manager = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        temp_weight = torch.empty((self.out_features, self.in_fseatures), dtype=torch.float32)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.has_bias:
            fanIn, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fanIn) if fanIn > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            temp_bias = torch.empty(self.out_features, dtype=torch.float32).unsqueeze(1)
            temp_weight = torch.cat((temp_weight, temp_bias), 1)
            self.in_features += 1
        
        self.weight = Parameter(temp_weight)

    def create_mat_mul_manager(self) -> None:
        self.mat_mul_manager = mmm.FixedPointMatMulManager(self.weight, self.weight_bit_width, self.input_bit_width)

    def eval(self):
        res = super().eval()
        self.create_mat_mul_manager()
        return res
    
    def forward(self, input:Tensor) -> Tensor:
        if self.has_bias:
            add_col = torch.ones((input.shape[0]), dtype=input.dtype, device=input.device)
            input = torch.cat((input, add_col))

        if self.training:
            return functional.linear(input, self.weight)
        else:
            return self.mat_mul_manager.mat_mul(input)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.hasBias
        )
    
