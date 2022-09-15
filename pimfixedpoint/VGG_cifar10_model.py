import math

import torch
import torch.nn as nn
import fixedPoint as fp
import fixedPoint.nn as fpnn
from fixedPoint.nn import torch_float


class VGG8B(nn.Module):
    def __init__(self):
        super(VGG8B, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 128, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 256, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Conv2d(256, 256, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 512, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Conv2d(512, 512, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Conv2d(512, 512, (3, 3), padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                  nn.Flatten(),
                                  nn.Linear(2048, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 10),
                                  nn.Dropout(p=0.2)
                                  )

    def forward(self, x):
        # conv layer
        x = self.conv(x)

        return x


class FixedPointVGG8B(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super(FixedPointVGG8B, self).__init__()
        self.device = device

        self.conv = \
            fpnn.MulInputSequential(fpnn.Conv2d([3, 32, 32], 128, (3, 3), padding=1),
                                    nn.ReLU(),
                                    fpnn.Conv2d([128, 32, 32], 256, (3, 3), padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([256, 16, 16], 256, (3, 3), padding=1),
                                    nn.ReLU(),
                                    fpnn.Conv2d([256, 16, 16], 512, (3, 3), padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([512, 8, 8], 512, (3, 3), padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([512, 4, 4], 512, (3, 3), padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    nn.Flatten(),
                                    fpnn.Quan(),
                                    fpnn.Linear(2048, 1024),
                                    fpnn.ReLU(),
                                    fpnn.Dropout(p=0.2),
                                    fpnn.Linear(1024, 10),
                                    fpnn.Dropout(p=0.2),
                                    fpnn.DeQuan()
                                    )

    def forward(self, x):
        # conv layer
        x = self.conv(x)

        return x


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, (3, 3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


VGG_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(batch_norm=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(VGG_cfg['A'], batch_norm=batch_norm))


def vgg13(batch_norm=False):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(VGG_cfg['B'], batch_norm=batch_norm))


def vgg16(batch_norm=False):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(VGG_cfg['D'], batch_norm=batch_norm))


def vgg19(batch_norm=False):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(VGG_cfg['E'], batch_norm=batch_norm))


class FixedPointVGG(nn.Module):
    """
    VGG model. different in dropout position
    """
    def __init__(self, features,
                 fc_output_bit_width=8,
                 fc_weight_bit_width=16,
                 fc_grad_output_bit_width=8,
                 fc_compute_weight_bit_width=8
                 ):
        super(FixedPointVGG, self).__init__()
        self.features = features
        self.classifier = fpnn.MulInputSequential(
            nn.Flatten(),
            fpnn.Quan(),
            fpnn.Linear(512, 512,
                        output_bit_width=fc_output_bit_width,
                        weight_bit_width=fc_weight_bit_width,
                        grad_output_bit_width=fc_grad_output_bit_width,
                        compute_weight_bit_width=fc_compute_weight_bit_width),
            fpnn.ReLU(),
            fpnn.Dropout(),
            fpnn.Linear(512, 512,
                        output_bit_width=fc_output_bit_width,
                        weight_bit_width=fc_weight_bit_width,
                        grad_output_bit_width=fc_grad_output_bit_width,
                        compute_weight_bit_width=fc_compute_weight_bit_width),
            fpnn.ReLU(),
            fpnn.Dropout(),
            fpnn.Linear(512, 10,
                        output_bit_width=fc_output_bit_width,
                        weight_bit_width=fc_weight_bit_width,
                        grad_output_bit_width=fc_grad_output_bit_width,
                        compute_weight_bit_width=fc_compute_weight_bit_width),
            fpnn.DeQuan()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fixed_point_make_layers(cfg,
                            conv_input_bit_width: int,
                            conv_output_bit_width: int,
                            conv_weight_bit_width: int,
                            conv_grad_output_bit_width: int,
                            conv_next_grad_output_bit_width: int,
                            conv_compute_weight_bit_width: int,
                            batch_norm: bool):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = fpnn.Conv2d(in_channels, v, (3, 3), padding=1,
                                 input_bit_width=conv_input_bit_width,
                                 output_bit_width=conv_output_bit_width,
                                 weight_bit_width=conv_weight_bit_width,
                                 grad_output_bit_width=conv_grad_output_bit_width,
                                 next_grad_output_bit_width=conv_next_grad_output_bit_width,
                                 compute_weight_bit_width=conv_compute_weight_bit_width,
                                 batch_norm=batch_norm
                                 )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, dtype=torch_float), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


def fp_vgg11(batch_norm=False):
    """VGG 11-layer model (configuration "A")"""
    return FixedPointVGG(fixed_point_make_layers(VGG_cfg['A'], batch_norm=batch_norm))


def fp_vgg16(conv_input_bit_width,
             conv_output_bit_width,
             conv_weight_bit_width,
             conv_grad_output_bit_width,
             conv_next_grad_output_bit_width,
             conv_compute_weight_bit_width,
             fc_output_bit_width,
             fc_weight_bit_width,
             fc_grad_output_bit_width,
             fc_compute_weight_bit_width,
             batch_norm=False):
    """VGG 16-layer model (configuration "D")"""
    return FixedPointVGG(
        fixed_point_make_layers(
            VGG_cfg['D'],
            conv_input_bit_width=conv_input_bit_width,
            conv_output_bit_width=conv_output_bit_width,
            conv_weight_bit_width=conv_weight_bit_width,
            conv_grad_output_bit_width=conv_grad_output_bit_width,
            conv_next_grad_output_bit_width=conv_next_grad_output_bit_width,
            conv_compute_weight_bit_width=conv_compute_weight_bit_width,
            batch_norm=batch_norm
        ),
        fc_output_bit_width=fc_output_bit_width,
        fc_weight_bit_width=fc_weight_bit_width,
        fc_grad_output_bit_width=fc_grad_output_bit_width,
        fc_compute_weight_bit_width=fc_compute_weight_bit_width
    )


def fp_vgg19(batch_norm=False):
    """VGG 19-layer model (configuration "E")"""
    return FixedPointVGG(fixed_point_make_layers(VGG_cfg['E'], batch_norm=batch_norm))
