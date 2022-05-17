import torch
import fixedPoint as fp
import torch.nn as nn
import torch.nn.functional as F
import fixedPoint.nn as fpnn
import fixedPoint.nn.pimFunction as fpF


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
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.quan = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.fc1 = fpnn.PimLinear(784, 128, batch_size, 16, 16, 16, device=device)
        self.fc2 = fpnn.PimLinear(128, 10, batch_size, 16, 16, 16, device=device)
        self.relu = fpF.PimRelu()
        self.dequan = fpF.DeQuanLayer(fp.half_data_flow_bit_width)

    def forward(self, x):
        x = self.flatten(x)
        x, x_p = self.quan(x)
        x, x_p = self.fc1(x, x_p)
        x, x_p = self.relu(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x = self.dequan(x, x_p)

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
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.conv = fpF.MulInputSequential(fpnn.PimConv2D([1, 28, 28], [5, 5], 10, batch_size, device=device),
                                           nn.ReLU(),
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                           fpnn.PimConv2D([10, 12, 12], [5, 5], 20, batch_size, device=device),
                                           nn.ReLU(),
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                           nn.Flatten(),
                                           fpF.QuanLayer(fp.half_data_flow_bit_width),
                                           fpF.FPDropout(dropout_ratio=0.2),
                                           fpnn.PimLinear(4 * 4 * 20, 10, batch_size, device=device),
                                           fpF.DeQuanLayer(fp.half_data_flow_bit_width))

    def forward(self, x):
        x = self.conv(x)

        return x


class FixedPointSimpleConvNet(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        self.conv = fpF.MulInputSequential(fpnn.PimConv2D([1, 28, 28], [5, 5], 30, batch_size, device=device),
                                           nn.ReLU(),
                                           nn.MaxPool2d(kernel_size=2, stride=2),
                                           nn.Flatten(),
                                           fpF.QuanLayer(fp.half_data_flow_bit_width),
                                           fpnn.PimLinear(12 * 12 * 30, 100, batch_size, device=device),
                                           fpF.PimRelu(),
                                           fpnn.PimLinear(100, 10, batch_size, device=device),
                                           fpF.DeQuanLayer(fp.half_data_flow_bit_width)
                                           )

    def forward(self, x):
        x = self.conv(x)

        return x


class VGG8B(nn.Module):
    def __init__(self):
        super(FixedPointVGG8B, self).__init__()

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
            fpF.MulInputSequential(fpnn.PimConv2D([3, 32, 32], [3, 3], 128, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   fpnn.PimConv2D([128, 32, 32], [3, 3], 256, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   fpnn.PimConv2D([256, 16, 16], [3, 3], 256, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   fpnn.PimConv2D([256, 16, 16], [3, 3], 512, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   fpnn.PimConv2D([512, 8, 8], [3, 3], 512, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   fpnn.PimConv2D([512, 4, 4], [3, 3], 512, batch_size, padding=1, device=device),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Flatten(),
                                   fpF.QuanLayer(fp.half_data_flow_bit_width),
                                   fpnn.PimLinear(2048, 1024, batch_size=batch_size, device=device),
                                   fpF.PimRelu(),
                                   fpF.FPDropout(dropout_ratio=0.2),
                                   fpnn.PimLinear(1024, 10, batch_size=batch_size, device=device),
                                   fpF.FPDropout(dropout_ratio=0.2),
                                   fpF.DeQuanLayer(fp.half_data_flow_bit_width)
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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
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

        self.conv1 = fpnn.PimConv2D([3, 32, 32], [3, 3], 64, batch_size, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = fpnn.PimConv2D([64, 32, 32], [3, 3], 64, batch_size, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = fpnn.PimConv2D([64, 16, 16], [3, 3], 128, batch_size, padding=1, device=device)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = fpnn.PimConv2D([128, 16, 16], [3, 3], 128, batch_size, padding=1, device=device)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = fpnn.PimConv2D([128, 8, 8], [3, 3], 256, batch_size, padding=1, device=device)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = fpnn.PimConv2D([256, 8, 8], [3, 3], 256, batch_size, padding=1, device=device)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = fpnn.PimConv2D([256, 4, 4], [3, 3], 512, batch_size, padding=1, device=device)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = fpnn.PimConv2D([512, 4, 4], [3, 3], 512, batch_size, padding=1, device=device)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = fpnn.PimConv2D([512, 2, 2], [3, 3], 512, batch_size, padding=1, device=device)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = fpnn.PimConv2D([512, 2, 2], [3, 3], 512, batch_size, padding=1, device=device)
        self.bn10 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()
        self.quan = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.dropout1 = fpF.FPDropout()
        self.fc1 = fpnn.PimLinear(512, 512, batch_size, device=device)
        self.relu1 = fpF.PimRelu()
        self.dropout2 = fpF.FPDropout()
        self.fc2 = fpnn.PimLinear(512, 512, batch_size, device=device)
        self.relu2 = fpF.PimRelu()
        self.fc3 = fpnn.PimLinear(512, 10, batch_size, device=device)
        self.dequan = fpF.DeQuanLayer(fp.half_data_flow_bit_width)

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
