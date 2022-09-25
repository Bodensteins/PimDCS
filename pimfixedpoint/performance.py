import torch.nn as nn
import torch

class AreaModule:
    def __init__(self, net) -> None:
        self.net = net

        self.PE_size = 0
        
        self.adc_area = 0.
        self.dac_area = 0.
        self.SH_area = 0.
        self.array_area = 0.
        self.adder_area = 0.

    def get_area_info(self, net) -> None:
        self.PE_size = self.get_PE_size(self, net)


    def get_PE_size(self, net) -> int:
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Linear):
                pass
            if isinstance(layer, nn.Conv2d):
                pass

        return self.PE_size
            

