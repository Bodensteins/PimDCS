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
        self.conv1 = nn.Conv2d(1, 10, (5, 5))
        self.conv2 = nn.Conv2d(10, 20, (5, 5))
        self.fc = nn.Linear(4 * 4 * 20, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.flatten(x)
        x = self.fc(x)

        return x


class PimConvMnist(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.quan1 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv1 = fpnn.PimConv2D([1, 28, 28], [5, 5], 10, batch_size, device=device)
        self.quan2 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv2 = fpnn.PimConv2D([10, 12, 12], [5, 5], 20, batch_size, device=device)
        self.fc = fpnn.PimLinear(4 * 4 * 20, 10, batch_size, device=device)
        self.flatten = nn.Flatten()
        self.quan3 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.dequan = fpF.DeQuanLayer(fp.half_data_flow_bit_width)

    def forward(self, x):
        x, x_f = self.quan1(x)
        _, _, x = self.conv1(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x, x_f = self.quan2(x)
        _, _, x = self.conv2(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.flatten(x)
        x, x_f = self.quan3(x)

        x, x_f = self.fc(x, x_f)
        x = self.dequan(x, x_f)

        return x


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(512, 1024, (3, 3), padding=1)

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # conv layer
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
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1) # this is merged in training

        return x


class PimVGG(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        self.quan1 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv1 = fpnn.PimConv2D([3, 32, 32], [3, 3], 128, batch_size, padding=1, device=device)
        self.relu1 = fpF.PimRelu()
        self.conv2 = fpnn.PimConv2D([128, 32, 32], [3, 3], 128, batch_size, padding=1, device=device)

        self.quan2 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv3 = fpnn.PimConv2D([128, 16, 16], [3, 3], 256, batch_size, padding=1, device=device)
        self.relu2 = fpF.PimRelu()
        self.conv4 = fpnn.PimConv2D([256, 16, 16], [3, 3], 256, batch_size, padding=1, device=device)

        self.quan3 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv5 = fpnn.PimConv2D([256, 8, 8], [3, 3], 512, batch_size, padding=1, device=device)
        self.relu3 = fpF.PimRelu()
        self.conv6 = fpnn.PimConv2D([512, 8, 8], [3, 3], 512, batch_size, padding=1, device=device)

        self.quan4 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.conv7 = fpnn.PimConv2D([512, 4, 4], [3, 3], 1024, batch_size, padding=1, device=device)

        self.flatten = nn.Flatten()
        self.quan5 = fpF.QuanLayer(fp.half_data_flow_bit_width)
        self.fc1 = fpnn.PimLinear(4096, 128, batch_size, device=device)
        self.relu4 = fpF.PimRelu()
        self.fc2 = fpnn.PimLinear(128, 10, batch_size, device=device)
        self.dequan = fpF.DeQuanLayer(fp.half_data_flow_bit_width)

    def forward(self, x):
        x, x_f = self.quan1(x)
        x, x_f, _ = self.conv1(x, x_f)
        x, x_f, _ = self.relu1(x, x_f)
        _, _, x = self.conv2(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x, x_f = self.quan2(x)
        x, x_f, _ = self.conv3(x, x_f)
        x, x_f, _ = self.relu2(x, x_f)
        _, _, x = self.conv4(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x, x_f = self.quan3(x)
        x, x_f, _ = self.conv5(x, x_f)
        x, x_f, _ = self.relu3(x, x_f)
        _, _, x = self.conv6(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x, x_f = self.quan4(x)
        _, _, x = self.conv7(x, x_f)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = torch.flatten(x, 1)
        x, x_f = self.quan5(x)
        x, x_f = self.fc1(x, x_f)
        x, x_f, _ = self.relu4(x, x_f)
        x, x_f = self.fc2(x, x_f)
        x = self.dequan(x, x_f)

        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
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

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # conv layer
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
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class ConvCifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, (3, 3))
        self.conv2 = nn.Conv2d(15, 75, (4, 4))
        self.conv3 = nn.Conv2d(75, 375, (3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 2 * 375, 400)
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x


class VGG8B(nn.Module):
    def __init__(self):
        super(VGG8B, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 256, (3, 3), padding=1)

        self.conv3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)

        self.conv5 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv6 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # conv layer
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
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
