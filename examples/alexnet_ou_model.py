import torch.nn as nn
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU
from examples.statistic import print_nn_Sequential_Modules_statistic

class PimAlexNet_OU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = nn.Sequential(
            Conv2d_OU(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), mm_manager_type=0),  # CIFAR-10 的图像为 32x32, 因此使用 3x3 卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_OU(64, 192, kernel_size=(3, 3), padding=(1, 1), mm_manager_type=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_OU(192, 384, kernel_size=(3, 3), padding=(1, 1), mm_manager_type=0),
            nn.ReLU(inplace=True),

            Conv2d_OU(384, 256, kernel_size=(3, 3), padding=(1, 1), mm_manager_type=0),
            nn.ReLU(inplace=True),

            Conv2d_OU(256, 256, kernel_size=(3, 3), padding=(1, 1), mm_manager_type=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Dropout(),
            Linear_OU(256 * 4 * 4, 1024, mm_manager_type=0),  # 根据 CIFAR-10 图像大小调整全连接层的输入维度
            nn.ReLU(inplace=True),

            nn.Dropout(),
            Linear_OU(1024, 512, mm_manager_type=0),
            nn.ReLU(inplace=True),

            Linear_OU(512, 10, mm_manager_type=0),  # 最后一层的输出节点数为类别数（10）
        )
        
    def forward(self, x):
        x = self.alexnet(x)
        return x
    
    def eval(self):
        res = super().eval()
        for module in self.alexnet.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.create_mat_mul_manager()
        return res
    
    def print_statistic(self):
        print_nn_Sequential_Modules_statistic(self.alexnet.modules)

    def reset_analyzer(self):
        for module in self.alexnet.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.data_analyzer.reset_input_statistic()
    