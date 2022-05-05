# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sgd import SGD


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


input = torch.randn([1, 10])
target = torch.randint(0, 10, [1])
net = TorchNet()
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9)
output = net.forward(input)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()

