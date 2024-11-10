import torch
import torch.nn as nn
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU
from examples.statistic import print_nn_Sequential_Modules_statistic

# 暂时不支持训练模式，去掉了辅助分类器
class PimGoogLeNet_OU(nn.Module):
    def __init__(self, num_classes=1000):
        super(PimGoogLeNet_OU, self).__init__()

        self.googlenet = nn.Sequential(
            self._make_conv1(),
            self._make_conv2(),
            self._make_inception3(),
            self._make_inception4(),
            self._make_inception5(),
            self._make_avgpool_and_fc(num_classes)
        )

    def forward(self, x):
        x = self.googlenet(x)
        return x

    def eval(self):
        res = super().eval()
        for module in self.googlenet.modules():
            if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
                module.create_mat_mul_manager()
        return res

    def print_statistic(self):
        print_nn_Sequential_Modules_statistic(self.googlenet.modules)

    def _make_conv1(self):
        return nn.Sequential(
            BasicConv2d(3, 64, kernel_size=(7,7), stride=2, padding=(3,3)),
            nn.MaxPool2d((3,3), stride=2, ceil_mode=True)
        )

    def _make_conv2(self):
        return nn.Sequential(
            BasicConv2d(64, 64, kernel_size=(1,1)),
            BasicConv2d(64, 192, kernel_size=(3,3), padding=(1,1)),
            nn.MaxPool2d((3,3), stride=2, ceil_mode=True)
        )

    def _make_inception3(self):
        return nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d((3,3), stride=2, ceil_mode=True)
        )

    def _make_inception4(self):
        return nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d((3,3), stride=2, ceil_mode=True)
        )


    def _make_inception5(self):
        return nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128)
        )

    def _make_avgpool_and_fc(self, num_classes):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            Linear_OU(1024, num_classes)
        )



class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1,1))

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1,1)),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3,3), padding=(1,1))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1,1)),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3,3), padding=(1,1))
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1,1))
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)




class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2d_OU(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x