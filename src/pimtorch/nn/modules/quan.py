from torch.autograd import Function
from torch.nn import Module
from .. import fixedPointArithmetic as fpA
from src.pimtorch.config.globalCfg import TensorType, globalCfg

class quan(Function):
    @staticmethod
    def forward(ctx, _input, bit_width):
        fp_output, fp_output_config = fpA.quantization_tensor(_input, bit_width, TensorType.Normal)
        return fpA.to_float(fp_output), fpA.to_float(fp_output_config)

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_config):
        # if debug_backward is True:
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        return fpA.de_quantization((fp_grad_output, fp_grad_output_config)), None


class dequan(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_cfg, back_bit_width: int):
        ctx.backBitWidth = back_bit_width

        return fpA.de_quantization((fp_input, fp_input_cfg))

    @staticmethod
    def backward(ctx, grad_output):
        # if debug_backward is True:
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)

        fp_grad_output, fp_grad_output_cfg = fpA.quantization_tensor(grad_output, ctx.backBitWidth, TensorType.Normal)

        return fpA.to_float(fp_grad_output), fpA.to_float(fp_grad_output_cfg), None


class Quan(Module):
    def __init__(self, bit_width=globalCfg.dataFlowBitWidth):
        super().__init__()
        self._bit_width = bit_width

    def forward(self, _input):
        return quan.apply(_input, self._bit_width)


class DeQuan(Module):
    def __init__(self, back_bit_width: int = globalCfg.dataFlowBitWidth):
        super().__init__()
        self.backBit = back_bit_width

    def forward(self, qinput, qinput_config):
        return dequan.apply(qinput, qinput_config, self.backBit)
