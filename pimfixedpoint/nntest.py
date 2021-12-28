from pimlinear import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sgd import SGD
from quantization import to_int
from pim_optimizer import PimSGD
from quantization import de_quantization

class PimNet(nn.Module):
    def __init__(self):
        super(PimNet, self).__init__()
        self.fc1 = PimLinear(10, 13)
        self.fc2 = PimLinear(13, 10)
        self.relu = PimRelu()
        self.dequan = DeQuanLayer()

    def forward(self, x, x_p):
        x, x_p = self.fc1(x, x_p)
        x, x_p = self.relu(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x = self.dequan(x, x_p)
        print(x)
        output = F.log_softmax(x, dim=1)
        return output


class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(10, 13)
        self.fc2 = nn.Linear(13, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        print("x={%s}"%x)
        output = F.log_softmax(x, dim=1)
        return output


input = torch.randn([1, 10], requires_grad=True)
input_nor_para = creat_quantization_para(bit_width=16, tensor_type=TensorType.Normal)
input_nor = quantization_tensor(input_nor_para, input, input.abs().max())
input_nor_para.requires_grad_()

target = torch.randint(0, 4, [1])
net = PimNet()
wArr1 = de_quantization([net.fc1.wArr, net.fc1.wArrConfig])
wArr2 = de_quantization([net.fc2.wArr, net.fc2.wArrConfig])
w1 = wArr1[:-1,:]

b1 = wArr1[-1, :]
w2 = wArr2[:-1,:]
b2 = wArr2[-1, :]
torch_net = TorchNet()
torch_net.fc1.weight.data = w1.t()
torch_net.fc1.bias.data = b1
torch_net.fc2.weight.data = w2.t()
torch_net.fc2.bias.data = b2

print("===== init =====")
print("fc1 weight")
print(torch_net.fc1.weight)
print("fc1 bias")
print(torch_net.fc1.bias)
print("fc2 weight")
print(torch_net.fc2.weight)
print("fc2 bias")
print(torch_net.fc2.bias)

optimizer = PimSGD(net, lr=0.1, momentum=0.9)
torch_optimizer = SGD(torch_net.parameters(), lr=0.1, momentum=0.9)

torch_output = torch_net.forward(input)
torch_loss = F.nll_loss(torch_output, target)
torch_loss.backward()
torch_optimizer.step()

print("===== torch opt =====")
print("fc1 weight")
print(torch_net.fc1.weight)
print("fc1 bias")
print(torch_net.fc1.bias)
print("fc2 weight")
print(torch_net.fc2.weight)
print("fc2 bias")
print(torch_net.fc2.bias)

output = net.forward(input_nor, input_nor_para)
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()