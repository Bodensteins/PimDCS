# -*- coding: utf-8 -*-

from fixedPoint.nn.modules.pimlinear import *
from fixedPoint.optim.pim_optimizer import *
import torch.nn as nn
import torch.nn.functional as F


class PimNet(nn.Module):
  def __init__(self):
    super(PimNet, self).__init__()
    self.fc1 = PimLinear(10, 10, 8, 2, 8)
    self.fc2 = PimLinear(10, 10, 8, 2, 8)
    self.relu = PimRelu()
    self.dequant = DeQuanLayer()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.dequant(x)
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


input = torch.randn([1, 10])
grad = torch.ones([1, 10])
input_nor = NormalTensor()
input_nor.fpA(input, input.abs().max(), 8)
grad_nor = NormalTensor()
grad_nor.fpA(grad, grad.abs().max(), 8)

if isinstance(input_nor, torch.Tensor):
  print("yes")

target = torch.randint(0, 10, [1])
net = PimNet()
optimizer = PimSGD(net, lr=0.1, momentum=0.9)
output = net.forward(input_nor)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()

