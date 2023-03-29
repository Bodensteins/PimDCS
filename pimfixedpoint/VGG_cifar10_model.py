import math

# import torch
import torch.nn as nn
# import fixedPoint as fp
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


# class FixedPointVGG8B(nn.Module):
#     def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
#         super(FixedPointVGG8B, self).__init__()
#         self.device = device
#
#         self.conv = \
#             fpnn.MulInputSequential(fpnn.Conv2d([3, 32, 32], 128, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     fpnn.Conv2d([128, 32, 32], 256, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
#                                     fpnn.Conv2d([256, 16, 16], 256, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     fpnn.Conv2d([256, 16, 16], 512, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
#                                     fpnn.Conv2d([512, 8, 8], 512, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
#                                     fpnn.Conv2d([512, 4, 4], 512, (3, 3), padding=1),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
#                                     nn.Flatten(),
#                                     fpnn.Quan(),
#                                     fpnn.Linear(2048, 1024),
#                                     fpnn.ReLU(),
#                                     fpnn.Dropout(p=0.2),
#                                     fpnn.Linear(1024, 10),
#                                     fpnn.Dropout(p=0.2),
#                                     fpnn.DeQuan()
#                                     )
#
#     def forward(self, x):
#         # conv layer
#         x = self.conv(x)
#
#         return x


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


def fp_vgg11(conv_input_bit_width,
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
    """VGG 11-layer model (configuration "A")"""
    return FixedPointVGG(fixed_point_make_layers(
        VGG_cfg['A'],
        conv_input_bit_width=conv_input_bit_width,
        conv_output_bit_width=conv_output_bit_width,
        conv_weight_bit_width=conv_weight_bit_width,
        conv_grad_output_bit_width=conv_grad_output_bit_width,
        conv_next_grad_output_bit_width=conv_next_grad_output_bit_width,
        conv_compute_weight_bit_width=conv_compute_weight_bit_width,
        batch_norm=batch_norm),
        fc_output_bit_width=fc_output_bit_width,
        fc_weight_bit_width=fc_weight_bit_width,
        fc_grad_output_bit_width=fc_grad_output_bit_width,
        fc_compute_weight_bit_width=fc_compute_weight_bit_width
    )


def fp_vgg13(conv_input_bit_width,
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
    """VGG 13-layer model (configuration "B")"""
    return FixedPointVGG(fixed_point_make_layers(
        VGG_cfg['B'],
        conv_input_bit_width=conv_input_bit_width,
        conv_output_bit_width=conv_output_bit_width,
        conv_weight_bit_width=conv_weight_bit_width,
        conv_grad_output_bit_width=conv_grad_output_bit_width,
        conv_next_grad_output_bit_width=conv_next_grad_output_bit_width,
        conv_compute_weight_bit_width=conv_compute_weight_bit_width,
        batch_norm=batch_norm),
        fc_output_bit_width=fc_output_bit_width,
        fc_weight_bit_width=fc_weight_bit_width,
        fc_grad_output_bit_width=fc_grad_output_bit_width,
        fc_compute_weight_bit_width=fc_compute_weight_bit_width
    )


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


def fp_vgg19(conv_input_bit_width,
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
    """VGG 19-layer model (configuration "E")"""
    return FixedPointVGG(
        fixed_point_make_layers(
            VGG_cfg['E'],
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


class VGG16ForMotivation(nn.Module):
    """
    VGG model
    """
    def __init__(self):
        super(VGG16ForMotivation, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(256, 256, (3, 3), padding=1)

        self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv13 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv1InputList = []
        self.conv2InputList = []
        self.conv3InputList = []
        self.conv4InputList = []
        self.conv5InputList = []
        self.conv6InputList = []
        self.conv7InputList = []
        self.conv8InputList = []
        self.conv9InputList = []
        self.conv10InputList = []
        self.conv11InputList = []
        self.conv12InputList = []
        self.conv13InputList = []

        self.fc1InputList = []
        self.fc2InputList = []
        self.fc3InputList = []

        self.conv1WeightList = []
        self.conv2WeightList = []
        self.conv3WeightList = []
        self.conv4WeightList = []
        self.conv5WeightList = []
        self.conv6WeightList = []
        self.conv7WeightList = []
        self.conv8WeightList = []
        self.conv9WeightList = []
        self.conv10WeightList = []
        self.conv11WeightList = []
        self.conv12WeightList = []
        self.conv13WeightList = []

        self.fc1WeightList = []
        self.fc2WeightList = []
        self.fc3WeightList = []

        self.logInterval = 1000
        self.index = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        if self.index == 0:
            # self.conv1WeightList.append(math.log2(max(self.conv1.weight.abs().max().item(),
            #                                           self.conv1.bias.abs().max().item())))
            self.conv2WeightList.append(math.log2(max(self.conv2.weight.abs().max().item(),
                                                      self.conv2.bias.abs().max().item())))
            self.conv3WeightList.append(math.log2(max(self.conv3.weight.abs().max().item(),
                                                      self.conv3.bias.abs().max().item())))
            # self.conv4WeightList.append(max(self.conv4.weight.abs().max(), self.conv4.bias.abs().max()))
            # self.conv5WeightList.append(max(self.conv5.weight.abs().max(), self.conv5.bias.abs().max()))
            # self.conv6WeightList.append(max(self.conv6.weight.abs().max(), self.conv6.bias.abs().max()))
            # self.conv7WeightList.append(max(self.conv7.weight.abs().max(), self.conv7.bias.abs().max()))
            # self.conv8WeightList.append(max(self.conv8.weight.abs().max(), self.conv8.bias.abs().max()))
            # self.conv9WeightList.append(max(self.conv9.weight.abs().max(), self.conv9.bias.abs().max()))
            # self.conv10WeightList.append(max(self.conv10.weight.abs().max(), self.conv10.bias.abs().max()))
            # self.conv11WeightList.append(max(self.conv11.weight.abs().max(), self.conv11.bias.abs().max()))
            # self.conv12WeightList.append(max(self.conv12.weight.abs().max(), self.conv12.bias.abs().max()))
            # self.conv13WeightList.append(max(self.conv13.weight.abs().max(), self.conv13.bias.abs().max()))
            # self.fc1WeightList.append(max(self.fc1.weight.abs().max(), self.fc1.bias.abs().max()))
            self.fc2WeightList.append(math.log2(max(self.fc2.weight.abs().max().item(),
                                                    self.fc2.bias.abs().max().item())))
            self.fc3WeightList.append(math.log2(max(self.fc3.weight.abs().max().item(),
                                                    self.fc3.bias.abs().max().item())))

        if self.index == 0:
            self.conv1InputList.append(max(x.abs().max(), 1))

        x = self.conv1(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv2InputList.append(math.log2(max(x.abs().max(), 1)))

        x = self.conv2(x)
        x = self.relu(x)

        x = self.pool(x)

        if self.index == 0:
            self.conv3InputList.append(math.log2(max(x.abs().max(), 1)))

        x = self.conv3(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv4InputList.append(math.log2(max(x.abs().max(), 1)))

        x = self.conv4(x)
        x = self.relu(x)

        x = self.pool(x)

        if self.index == 0:
            self.conv5InputList.append(max(x.abs().max(), 1))

        x = self.conv5(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv6InputList.append(max(x.abs().max(), 1))

        x = self.conv6(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv7InputList.append(max(x.abs().max(), 1))

        x = self.conv7(x)
        x = self.relu(x)

        x = self.pool(x)

        if self.index == 0:
            self.conv8InputList.append(max(x.abs().max(), 1))

        x = self.conv8(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv9InputList.append(max(x.abs().max(), 1))

        x = self.conv9(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv10InputList.append(max(x.abs().max(), 1))

        x = self.conv10(x)
        x = self.relu(x)

        x = self.pool(x)

        if self.index == 0:
            self.conv11InputList.append(max(x.abs().max(), 1))

        x = self.conv11(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv12InputList.append(max(x.abs().max(), 1))

        x = self.conv12(x)
        x = self.relu(x)

        if self.index == 0:
            self.conv13InputList.append(max(x.abs().max(), 1))

        x = self.conv13(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)

        x = self.dropout(x)

        if self.index == 0:
            self.fc1InputList.append(max(x.abs().max(), 1))

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        if self.index == 0:
            self.fc2InputList.append(math.log2(max(x.abs().max(), 1)))

        x = self.fc2(x)
        x = self.relu(x)

        if self.index == 0:
            self.fc3InputList.append(math.log2(max(x.abs().max(), 1)))

        x = self.fc3(x)

        self.index = (self.index + 1) % 1000

        return x
