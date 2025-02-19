import math
import torch
from torch import Tensor
# from torch.autograd import Function
from torch.nn import Module, Parameter, init, functional
from src.pimtorch.config.globalCfg import globalCfg
from src.pimtorch.nn import matMulManager as mmm
from src.pimtorch.nn import fixedPointDataAnalyze as fpDA


# 仅支持前向传播，不考虑训练
# 已测试
class Linear_OU(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_bit_width: int = globalCfg.dataFlowBitWidth, 
                 output_bit_width: int = globalCfg.dataFlowBitWidth, weight_bit_width: int = globalCfg.dataFlowBitWidth,
                 grad_output_bit_width: int = globalCfg.dataFlowBitWidth) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.input_bit_width = input_bit_width
        self.output_bit_width = output_bit_width
        self.weight_bit_width = weight_bit_width
        self.grad_output_bit_width = grad_output_bit_width

        self.weight = Parameter(torch.empty((out_features, in_features), dtype=torch.float))
        if self.has_bias:
            self.bias = Parameter(torch.empty(out_features, dtype=torch.float))
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
        # print("Linear_OU create mmm")
        weight_t = self.weight.t()
        if self.has_bias:
            weight_t = torch.cat((weight_t, self.bias.unsqueeze(0)), 0)
        # self.mm_manager = mmm.FixedPointMatMulManager(weight_t, self.weight_bit_width, self.input_bit_width)
        self.mm_manager = mmm.FixedPointMatMulManager_OU(weight_t, self.weight_bit_width, self.input_bit_width)
        # self.mm_manager = mmm.FixedPointMatMulManager_OU_PN(weight_t, self.weight_bit_width, self.input_bit_width)
        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
            if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU_PN):
                self.data_analyzer = fpDA.FpDataAnalyzer_PN(self.mm_manager)
            else:
                self.data_analyzer = fpDA.FpDataAnalyzer(self.mm_manager)
            self.mm_manager.set_data_analyzer(self.data_analyzer)
            self.data_analyzer.update_all_weight_statistic()
    
    def forward(self, input:Tensor) -> Tensor:
        if self.training:
            # print("train Linear_OU")
            return functional.linear(input, self.weight, self.bias)
        else:
            # print("mutiply with mm_manager in Linear_OU")
            if self.has_bias:
                add_col = torch.ones((input.shape[0]), dtype=input.dtype, device=input.device).unsqueeze(1)
                input = torch.cat((input, add_col), dim=1)
                
            # mm_res = self.mm_manager.mat_mul(input)
            output = self.mm_manager.fake_mat_mul(input)

            if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
                self.data_analyzer.update_all_input_statistic()
            if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU_PN):
                self.data_analyzer.update_ou_column_output_num()

            return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.has_bias)
    
    def print_statistic(self):
        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_OU):
            self.data_analyzer.print_statistic()

    def save_statistic(self):
        pass
