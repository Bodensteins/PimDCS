import math
import torch
from torch import Tensor
# from torch.autograd import Function
from torch.nn import Module, Parameter, init, functional
from config.globalCfg import globalCfg
from src.nn_modules import matMulManager as mmm
from src.nn_modules import dumpPartialSum as dps


# 仅支持前向传播，不考虑训练
class FpLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_bit_width: int = globalCfg.dataFlowBitWidth, 
                 output_bit_width: int = globalCfg.dataFlowBitWidth, weight_bit_width: int = globalCfg.dataFlowBitWidth,
                 grad_output_bit_width: int = globalCfg.dataFlowBitWidth, mm_manager_type: int = globalCfg.mmManagerTpye,
                 layer_no: int = 0, n_noise_max: int = 11) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.input_bit_width = input_bit_width
        self.output_bit_width = output_bit_width
        self.weight_bit_width = weight_bit_width
        self.grad_output_bit_width = grad_output_bit_width
        self.mm_manager_type = mm_manager_type

        self.weight = Parameter(torch.empty((out_features, in_features), dtype=torch.float))
        if self.has_bias:
            self.bias = Parameter(torch.empty(out_features, dtype=torch.float))
        else:
            self.bias = None
        self.mm_manager = None
        self.data_analyzer = None

        self.n_noise_max = n_noise_max
        self.layer_no = layer_no

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fanIn, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fanIn) if fanIn > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def create_mat_mul_manager(self) -> None:
        weight_t = self.weight.t()
        if self.has_bias:
            weight_t = torch.cat((weight_t, self.bias.unsqueeze(0)), 0)
            
        if self.mm_manager_type == 4:
            self.mm_manager = mmm.FixedPointMatMulManager_Simp(weight_t, self.weight_bit_width, self.input_bit_width,
                                                                    layer_no=self.layer_no)
        elif self.mm_manager_type == 5:
            self.mm_manager = mmm.FixedPointMatMulManager_TRQ(weight_t, self.weight_bit_width, self.input_bit_width, 
                                                                  layer_no=self.layer_no)
        elif self.mm_manager_type == 6:
            self.mm_manager = mmm.FixedPointMatMulManager_Spec(weight_t, self.weight_bit_width, self.input_bit_width,
                                                                    layer_no=self.layer_no)
        elif self.mm_manager_type == 7:
            self.mm_manager = mmm.FixedPointMatMulManager_Tailor(weight_t, self.weight_bit_width, self.input_bit_width,
                                                                        n_noise_max=self.n_noise_max)
        else:
            self.mm_manager = mmm.FixedPointMatMulManager(weight_t, self.weight_bit_width, self.input_bit_width)

        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_Spec):
            self.data_analyzer = dps.FpDataAnalyzer_Spec(self.mm_manager)
            self.mm_manager.set_data_analyzer(self.data_analyzer)
        elif isinstance(self.mm_manager, mmm.FixedPointMatMulManager_TRQ):
            self.data_analyzer = dps.FpDataAnalyzer_TRQ(self.mm_manager)
            self.mm_manager.set_data_analyzer(self.data_analyzer)
        elif isinstance(self.mm_manager, mmm.FixedPointMatMulManager_Simp):
            self.data_analyzer = dps.FpDataAnalyzer_Spec(self.mm_manager)
            self.mm_manager.set_data_analyzer(self.data_analyzer)
    
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
            return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.has_bias)
    
    def set_layer_no(self, layer_no: int = 0):
        self.layer_no = self.mm_manager.layer_no = layer_no

    def save_statistic(self, save_path:str=None, postfix:str='test'):
        if isinstance(self.mm_manager, mmm.FixedPointMatMulManager_Spec):
            self.data_analyzer.save_statistic(save_path, postfix)
        elif isinstance(self.mm_manager, mmm.FixedPointMatMulManager_TRQ):
            self.data_analyzer.save_statistic(save_path, postfix)
        elif isinstance(self.mm_manager, mmm.FixedPointMatMulManager_Simp):
            self.data_analyzer.save_statistic(save_path, postfix)

