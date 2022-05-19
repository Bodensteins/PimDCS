from torch.nn import Module
from .. import functional as fpF


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout_ratio = p

    def forward(self, fp_input, fp_input_cfg):
        return fpF.dropout.apply(fp_input, fp_input_cfg, self.dropout_ratio, self.training)
    