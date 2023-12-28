from torch.nn import Module
from torch.autograd import Function
from .. import fixedPointArithmetic as fpA

class relu(Function):
    @staticmethod
    def forward(ctx, qinput, qinput_config):
        neg_position = fpA.fixed_point_less((qinput, qinput_config), 0)
        qinput_ = qinput.clone()
        qinput_[neg_position] = 0  # The zero in ieee754 is all zero as well.
        ctx.save_for_backward(neg_position)
        return qinput_, qinput_config

    @staticmethod
    def backward(ctx, qgrad_output, qgrad_output_config):
        # if debug_backward is True:
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        neg_position, = ctx.saved_tensors
        qgrad_output[neg_position] = 0

        return qgrad_output, qgrad_output_config

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput, qinput_config):
        return relu.apply(qinput, qinput_config)