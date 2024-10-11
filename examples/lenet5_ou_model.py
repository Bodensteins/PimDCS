import torch.nn as nn
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU

class PimLeNet5_OU(nn.Module):
    def __init__(self):
        super().__init__()
        self.lenet5 = nn.Sequential(
            Conv2d_OU(1, 6, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Conv2d_OU(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Conv2d_OU(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),
            
            nn.Flatten(),
            Linear_OU(120, 84),
            nn.ReLU(),
            Linear_OU(84, 10)
        )
        
    def forward(self, x):
        x = self.lenet5(x)
        return x
    
    def eval(self):
        res = super().eval()
        for module in self.lenet5.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.create_mat_mul_manager()
        return res
    
    def print_statistic(self):
        for module in self.lenet5.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                print("---------------------")
                print(module)
                module.print_statistic()
                print("---------------------\n")   

    def reset_analyzer(self):
        for module in self.lenet5.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.data_analyzer.reset_input_statistic()
    