import torch
from torch.autograd import Function
from torch.nn import Module
from src.pimtorch.config.globalCfg import globalCfg
from .. import fixedPointArithmetic as fpA

class dropout(Function):
    @staticmethod
    def forward(ctx, fp_input, fp_input_cfg, dropout_ratio, training):
        if training:
            mask = torch.rand_like(fp_input) < dropout_ratio  # position of smaller than dropout_ratio
            ctx.save_for_backward(mask)  # tensor should be saved in save for backward
            fp_input[mask] = 0  # The zero in ieee754 is all zero as well.
            return fp_input, fp_input_cfg
        else:
            int_input = fpA.to_int(fp_input)
            int_input = int_input.mul(1 - dropout_ratio).to(dtype=globalCfg.torchInt)

            return fpA.to_float(int_input), fp_input_cfg

    @staticmethod
    def backward(ctx, fp_grad_output, fp_grad_output_cfg):
        # if debug_backward is True:
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        mask, = ctx.saved_tensors
        fp_grad_output[mask] = 0
        return fp_grad_output, fp_grad_output_cfg, None, None


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout_ratio = p

    def forward(self, fp_input, fp_input_cfg):
        return dropout.apply(fp_input, fp_input_cfg, self.dropout_ratio, self.training)
    