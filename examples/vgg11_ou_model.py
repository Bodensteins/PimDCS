import torch.nn as nn
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU
from examples.statistic import print_nn_Sequential_Modules_statistic

class PimVGG11_OU(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.totalDecodeTimes = 0
        self.vgg11 = nn.Sequential(Conv2d_OU(3, 64, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   Conv2d_OU(64, 128, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   Conv2d_OU(128, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   Conv2d_OU(256, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   Conv2d_OU(256, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   Conv2d_OU(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   Conv2d_OU(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   Conv2d_OU(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Flatten(),
                                   Linear_OU(1 * 1 * 512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   Linear_OU(512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   Linear_OU(512, num_classes)
                                   )
      

    def forward(self, x):
        x = self.vgg11(x)
        return x

    def eval(self):
        res = super().eval()
        for module in self.vgg11.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.create_mat_mul_manager()
        return res
    
    def print_statistic(self):
        print_nn_Sequential_Modules_statistic(self.vgg11.modules)

    def reset_analyzer(self):
        for module in self.vgg11.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.data_analyzer.reset_input_statistic()