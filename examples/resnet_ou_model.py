import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU

def fixed_point_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
                        dilation: int = 1) -> Conv2d_OU:
    """3x3 convolution with padding"""
    return Conv2d_OU(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                       padding=dilation, groups=groups, bias=False, dilation=(dilation, dilation), )


def fixed_point_conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d_OU:
    """1x1 convolution"""
    return Conv2d_OU(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False)


class BasicBlock_OU(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock_OU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = fixed_point_conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels, dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = fixed_point_conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels, dtype=torch.float)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def create_mat_mul_manager(self):
        self.conv1.create_mat_mul_manager()
        self.conv2.create_mat_mul_manager()
    
    def print_statistic(self):
        print("---------------------")
        print(self.conv1)
        self.conv1.print_statistic()
        print("---------------------\n")
        print("---------------------")
        print(self.conv2)
        self.conv2.print_statistic()
        print("---------------------\n")
        
    def reset_analyzer(self):
        self.conv1.data_analyzer.reset_input_statistic()
        self.conv2.data_analyzer.reset_input_statistic()


class Bottleneck_OU(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck_OU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = fixed_point_conv1x1(in_channels, width)
        self.bn1 = norm_layer(width, dtype=torch.float)
        self.conv2 = fixed_point_conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, dtype=torch.float)
        self.conv3 = fixed_point_conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion, dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def create_mat_mul_manager(self):
        # print("create")
        self.conv1.create_mat_mul_manager()
        self.conv2.create_mat_mul_manager()
        self.conv3.create_mat_mul_manager()
    
    def print_statistic(self):
        print("---------------------")
        print(self.conv1)
        self.conv1.print_statistic()
        print("---------------------\n")
        print("---------------------")
        print(self.conv2)
        self.conv2.print_statistic()
        print("---------------------\n")
        print("---------------------")
        print(self.conv3)
        self.conv3.print_statistic()
        print("---------------------\n")
        
    def reset_analyzer(self):
        self.conv1.data_analyzer.reset_input_statistic()
        self.conv2.data_analyzer.reset_input_statistic()
        self.conv3.data_analyzer.reset_input_statistic()


class PimResNet_OU(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock_OU, Bottleneck_OU]],
            layers: List[int],
            num_classes: int = 1000,  # for ImageNet
            # num_classes: int = 10,  # for cifar10
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(PimResNet_OU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d_OU(3, self.in_channels, kernel_size=(7, 7), stride=(2, 2), padding=3,
                                 bias=False)
        self.bn1 = norm_layer(self.in_channels, dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.fixed_point_make_layer(block, 64, layers[0])
        self.layer2 = self.fixed_point_make_layer(block, 128, layers[1], stride=2,
                                                  dilate=replace_stride_with_dilation[0])
        self.layer3 = self.fixed_point_make_layer(block, 256, layers[2], stride=2,
                                                  dilate=replace_stride_with_dilation[1])
        self.layer4 = self.fixed_point_make_layer(block, 512, layers[3], stride=2,
                                                  dilate=replace_stride_with_dilation[2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear_OU(512 * block.expansion, num_classes)

        for m in self.modules():
            # if isinstance(m, Conv2d_OU):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_OU):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock_OU):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def fixed_point_make_layer(self, block: Type[Union[BasicBlock_OU, Bottleneck_OU]], channels: int,
                               blocks: int,
                               stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                fixed_point_conv1x1(self.in_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion, dtype=torch.float),
            )

        layers = [block(self.in_channels, channels, stride, downsample, self.groups, self.base_width,
                        previous_dilation, norm_layer)]

        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def eval(self):
        res = super().eval()
        self.conv1.create_mat_mul_manager()
        for block in self.layer1.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.create_mat_mul_manager()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.create_mat_mul_manager()
        for block in self.layer2.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.create_mat_mul_manager()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.create_mat_mul_manager()
        for block in self.layer3.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.create_mat_mul_manager()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.create_mat_mul_manager()
        for block in self.layer4.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.create_mat_mul_manager()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.create_mat_mul_manager()
        self.fc.create_mat_mul_manager()
        return res

    def print_statistic(self):
        print("---------------------")
        print(self.conv1)
        self.conv1.print_statistic()
        print("---------------------\n")
        
        for block in self.layer1.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.print_statistic()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.print_statistic()
                
        for block in self.layer2.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.print_statistic()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.print_statistic()
                
        for block in self.layer3.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.print_statistic()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.print_statistic()
                
        for block in self.layer4.modules():
            if isinstance(block, Conv2d_OU) or isinstance(block, BasicBlock_OU) or isinstance(block, Bottleneck_OU):
                block.print_statistic()
            if isinstance(block, nn.Sequential):
                for m in block.modules():
                    if isinstance(m, BasicBlock_OU) or isinstance(m, Bottleneck_OU):
                        m.print_statistic()

        print("---------------------")
        print(self.fc)
        self.fc.print_statistic()
        print("---------------------\n")
        

def resnet_ou(
        # arch: str,
        block: Type[Union[BasicBlock_OU, Bottleneck_OU]],
        layers: List[int],
        pretrained: bool,
        # progress: bool,
        **kwargs: Any
) -> PimResNet_OU:
    model = PimResNet_OU(block, layers, **kwargs)
    if pretrained:
        raise Exception("Sorry, we don't support pretrained model!")
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet18_ou(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> PimResNet_OU:
    return resnet_ou('resnet18', BasicBlock_OU, [2, 2, 2, 2], pretrained, progress,
                              **kwargs)


def resnet50_ou(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> PimResNet_OU:
    return resnet_ou('resnet50', Bottleneck_OU, [3, 4, 6, 3], pretrained, progress,
                              **kwargs)


def resnet152_ou(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> PimResNet_OU:
    return resnet_ou('resnet152', Bottleneck_OU, [3, 8, 36, 3], pretrained, progress,
                              **kwargs)
