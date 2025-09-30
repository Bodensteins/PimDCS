import math
import torch.nn as nn

class VGG_cifar10(nn.Module):
    """
    VGG model
    """
    def __init__(self, features):
        super(VGG_cifar10, self).__init__()
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


class VGG_imagenet(nn.Module):
    def __init__(self, features):
        super(VGG_imagenet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def eval(self):
        res = super().eval()
        return res

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


def vgg11_imagenet(batch_norm=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG_imagenet(make_layers(VGG_cfg['A'], batch_norm=batch_norm))

def vgg11_cifar10(batch_norm=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG_cifar10(make_layers(VGG_cfg['A'], batch_norm=batch_norm))

def vgg13_imagenet(batch_norm=False):
    """VGG 13-layer model (configuration "B")"""
    return VGG_imagenet(make_layers(VGG_cfg['B'], batch_norm=batch_norm))


def vgg16_imagenet(batch_norm=False):
    """VGG 16-layer model (configuration "D")"""
    return VGG_imagenet(make_layers(VGG_cfg['D'], batch_norm=batch_norm))


def vgg1_imagenet(batch_norm=False):
    """VGG 19-layer model (configuration "E")"""
    return VGG_imagenet(make_layers(VGG_cfg['E'], batch_norm=batch_norm))
