from pimlinear import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sgd import SGD
from quantization import float_to_int


class PimNet(nn.Module):
    def __init__(self):
        super(PimNet, self).__init__()
        self.fc1 = PimLinear(10, 10)
        self.fc2 = PimLinear(10, 10)
        self.relu = PimRelu()
        self.dequan = DeQuanLayer()

    def forward(self, x, x_p):
        x, x_p = self.fc1(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x, x_p= self.relu(x, x_p)
        x = self.dequan(x, x_p)
        output = F.log_softmax(x, dim=1)
        return output


class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# net = PimNet()
# for layer in net.named_modules():
#   if isinstance(layer[1], PimLinear):
#     print(layer)
# for param in net.parameters():
#   print(param)


input = torch.randn([1, 10], requires_grad=True)
input_nor_para = creat_quantization_para(bit_width=8, tensor_type=TensorType.Normal)
input_nor = quantization_tensor(input_nor_para, input, input.abs().max())

input_nor_para.requires_grad_()

target = torch.randint(0, 10, [1])
net = PimNet()
output = net.forward(input_nor, input_nor_para)
loss = F.nll_loss(output, target)
loss.backward()
