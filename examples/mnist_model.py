import torch
import torch.nn as nn
import torch.nn.functional as F
import src.pimtorch.nn as fpnn
from src.pimtorch.nn.modules.linear_ou import Linear_OU
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU


class PimFcMnist_OU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = Linear_OU(784, 128)
        self.fc2 = Linear_OU(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

    def eval(self):
        res = super().eval()
        self.fc1.create_mat_mul_manager()
        self.fc2.create_mat_mul_manager()
        return res
    
    def print_statistic(self):
        print("---------fc1---------")
        self.fc1.print_statistic()
        print("---------------------\n")
        print("---------fc2---------")
        self.fc2.print_statistic()
        print("---------------------\n")

    def reset_analyzer(self):
        self.fc1.data_analyzer.reset_input_statistic()
        self.fc2.data_analyzer.reset_input_statistic()


class PimConvMnist_OU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(Conv2d_OU(1, 10, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  Conv2d_OU(10, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Flatten(),
                                  nn.Dropout(p=0.2),
                                  Linear_OU(4 * 4 * 20, 10)
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def eval(self):
        res = super().eval()
        for module in self.conv.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.create_mat_mul_manager()
        return res

    def print_statistic(self):
        for module in self.conv.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                print("---------------------")
                print(module)
                module.print_statistic()
                print("---------------------\n")   

    def reset_analyzer(self):
        for module in self.conv.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.data_analyzer.reset_input_statistic()


class FcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class PimFcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.quan = fpnn.Quan()
        self.fc1 = fpnn.Linear(784, 128)
        self.fc2 = fpnn.Linear(128, 10)
        self.relu = fpnn.ReLU()
        self.dequan = fpnn.DeQuan()

    def forward(self, x):
        x = self.flatten(x)
        x, x_p = self.quan(x)
        x, x_p = self.fc1(x, x_p)
        x, x_p = self.relu(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x = self.dequan(x, x_p)

        return x


class PimDeepFcMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.deepFc = fpnn.MulInputSequential(nn.Flatten(),
                                              fpnn.Quan(),
                                              fpnn.Linear(784, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 128),
                                              fpnn.ReLU(),
                                              fpnn.Linear(128, 10),
                                              fpnn.DeQuan())

    def forward(self, x):
        x = self.deepFc(x)

        return x


class ConvMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 10, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Conv2d(10, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Flatten(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(4 * 4 * 20, 10)
                                  )

    def forward(self, x):
        x = self.conv(x)

        return x


class PimConvMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = fpnn.MulInputSequential(fpnn.Conv2d(1, 10, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                            fpnn.Conv2d(10, 20, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                            nn.Flatten(),
                                            fpnn.Quan(),
                                            fpnn.Dropout(p=0.2),
                                            fpnn.Linear(4 * 4 * 20, 10),
                                            fpnn.DeQuan())

    def forward(self, x):
        x = self.conv(x)

        return x


class FixedPointSimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = fpnn.MulInputSequential(fpnn.Conv2d(1, 30, (5, 5)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=2, stride=2),
                                            nn.Flatten(),
                                            fpnn.Quan(),
                                            fpnn.Linear(12 * 12 * 30, 100),
                                            fpnn.ReLU(),
                                            fpnn.Linear(100, 10),
                                            fpnn.DeQuan())

    def forward(self, x):
        x = self.conv(x)

        return x
