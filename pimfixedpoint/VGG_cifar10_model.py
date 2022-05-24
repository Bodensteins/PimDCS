import torch
import torch.nn as nn
import torch.nn.functional as F
import fixedPoint as fp
import fixedPoint.nn as fpnn


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
            fpnn.MulInputSequential(fpnn.Conv2d([3, 32, 32], 128, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    fpnn.Conv2d([128, 32, 32], 256, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([256, 16, 16], 256, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    fpnn.Conv2d([256, 16, 16], 512, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([512, 8, 8], 512, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    fpnn.Conv2d([512, 4, 4], 512, (3, 3), batch_size, padding=1, device=device),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    nn.Flatten(),
                                    fpnn.Quan(fp.half_data_flow_bit_width),
                                    fpnn.Linear(2048, 1024, batch_size=batch_size, device=device),
                                    fpnn.ReLU(),
                                    fpnn.Dropout(p=0.2),
                                    fpnn.Linear(1024, 10, batch_size=batch_size, device=device),
                                    fpnn.Dropout(p=0.2),
                                    fpnn.DeQuan(fp.half_data_flow_bit_width)
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


class FixedPointVGG13(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super(FixedPointVGG13, self).__init__()
        self.device = device

        # self.conv1 = fpnn.Conv2D([3, 32, 32], [3, 3], 64, batch_size, padding=1, device=device)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = fpnn.Conv2D([64, 32, 32], [3, 3], 64, batch_size, padding=1, device=device)
        # self.bn2 = nn.BatchNorm2d(64)
        #
        # self.conv3 = fpnn.Conv2D([64, 16, 16], [3, 3], 128, batch_size, padding=1, device=device)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = fpnn.Conv2D([128, 16, 16], [3, 3], 128, batch_size, padding=1, device=device)
        # self.bn4 = nn.BatchNorm2d(128)
        #
        # self.conv5 = fpnn.Conv2D([128, 8, 8], [3, 3], 256, batch_size, padding=1, device=device)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.conv6 = fpnn.Conv2D([256, 8, 8], [3, 3], 256, batch_size, padding=1, device=device)
        # self.bn6 = nn.BatchNorm2d(256)
        #
        # self.conv7 = fpnn.Conv2D([256, 4, 4], [3, 3], 512, batch_size, padding=1, device=device)
        # self.bn7 = nn.BatchNorm2d(512)
        # self.conv8 = fpnn.Conv2D([512, 4, 4], [3, 3], 512, batch_size, padding=1, device=device)
        # self.bn8 = nn.BatchNorm2d(512)
        #
        # self.conv9 = fpnn.Conv2D([512, 2, 2], [3, 3], 512, batch_size, padding=1, device=device)
        # self.bn9 = nn.BatchNorm2d(512)
        # self.conv10 = fpnn.Conv2D([512, 2, 2], [3, 3], 512, batch_size, padding=1, device=device)
        # self.bn10 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()
        self.quan = fpnn.Quan(fp.half_data_flow_bit_width)
        self.dropout1 = fpnn.Dropout()
        self.fc1 = fpnn.Linear(512, 512, batch_size, device=device)
        self.relu1 = fpnn.ReLU()
        self.dropout2 = fpnn.Dropout()
        self.fc2 = fpnn.Linear(512, 512, batch_size, device=device)
        self.relu2 = fpnn.ReLU()
        self.fc3 = fpnn.Linear(512, 10, batch_size, device=device)
        self.dequan = fpnn.DeQuan(fp.half_data_flow_bit_width)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = torch.flatten(x, 1)
        x, x_f = self.quan(x)
        x, x_f = self.dropout1(x, x_f)
        x, x_f = self.fc1(x, x_f)
        x, x_f = self.relu1(x, x_f)
        x, x_f = self.dropout2(x, x_f)
        x, x_f = self.fc2(x, x_f)
        x, x_f = self.relu2(x, x_f)
        x, x_f = self.fc3(x, x_f)
        x = self.dequan(x, x_f)

        return x


class FixedPointVGG(nn.Module):
    """
    VGG model
    """
    def __init__(self, features, batch_size, device):
        super(FixedPointVGG, self).__init__()
        self.features = features
        self.classifier = fpnn.MulInputSequential(
            nn.Flatten(),
            fpnn.Quan(fp.half_data_flow_bit_width),
            fpnn.Linear(512, 512, batch_size=batch_size, device=device),
            fpnn.ReLU(),
            fpnn.Dropout(),
            fpnn.Linear(512, 512, batch_size=batch_size, device=device),
            fpnn.ReLU(),
            fpnn.Dropout(),
            fpnn.Linear(512, 10, batch_size=batch_size, device=device),
            fpnn.DeQuan(fp.half_data_flow_bit_width)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fixed_point_make_layers(cfg, batch_size, device, batch_norm=False):
    layers = []
    in_channels = 3
    height = 32
    width = 32
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            height = height // 2
            width = width // 2
        else:
            conv2d = fpnn.Conv2d([in_channels, height, width], v, (3, 3),
                                 batch_size=batch_size, padding=1, device=device)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


def fp_vgg19(batch_size, device, batch_norm=False):
    """VGG 19-layer model (configuration "E")"""
    return FixedPointVGG(fixed_point_make_layers(VGG_cfg['E'],
                                                 batch_size=batch_size, device=device, batch_norm=batch_norm),
                         batch_size=batch_size, device=device)
