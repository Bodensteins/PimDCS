import torch.nn as nn
from .. import functional as fpF


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qinput, qinput_config):
        return fpF.relu.apply(qinput, qinput_config)