# -*- coding: utf-8 -*-
import math
import torch
from torch.nn import init
from torch import Tensor
from pimtorch import SimpleLogicArray, PimArrayType

class PimLinearFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, wb, wb_t, prev, is_training, input, weight, bias = None):
    ctx.save_for_backward(input, weight, bias, wb, wb_t, prev)
    if is_training and weight.requires_grad:
      prev.write_mat(input)
      if bias is not None:
        wb.write_mat(torch.cat([weight.t(), bias.unsqueeze(0)], 0))
      else:
        wb.write_mat(weight.t())

    pim_input = input
    if bias is not None:
      m = torch.nn.ConstantPad2d((0, 1, 0, 0), 1)
      pim_input = m(input)
    pim_output = wb.mm(pim_input)
    return pim_output

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, bias, wb, wb_t, prev = ctx.saved_variables
    pim_grad_input = wb_t.mm(grad_output)
    pim_grad_weight = prev.mm(grad_output.t())
    pim_grad_bias = None
    if bias is not None:
      pim_grad_bias = grad_output.sum(0)

    return pim_grad_input, pim_grad_weight, pim_grad_bias


class PimLinear(torch.nn.Module):
  def __init__(self, in_features, out_features, batch_size, pim_type, bias = True):
    super(PimLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.batch_size = batch_size
    self.pim_type = pim_type
    self.is_training = True
    self.weight = torch.nn.Parameter(Tensor(out_features, in_features))
    if bias is not None:
      self.wb = SimpleLogicArray(in_features + 1, out_features)
      self.bias = torch.nn.Parameter(Tensor(out_features))
    else:
      self.wb = SimpleLogicArray(in_features, out_features)
      self.register_parameter('bias', None)
    self.wb_t = SimpleLogicArray(out_features, in_features)
    self.prev = SimpleLogicArray(batch_size, in_features)

    # self.register_parameter('wb', self.wb.read_mat())
    # self.register_parameter('wb_t', self.wb_t.read_mat())
    # self.register_parameter('prev', self.prev.read_mat())

    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      init.uniform_(self.bias, -bound, bound)

  def forward(self, input: Tensor) -> Tensor:
    return PimLinearFunction.apply(self.wb, self.wb_t, self.prev, self.is_training, input, self.weight, self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None
    )

  def train(self, mode: bool = True):
    if mode is not True:
      self.sync_weight()
    self.is_training = mode

  def sync_weight(self):
    if self.bias is not None:
      self.wb.write_mat(torch.cat([self.weight.t(), self.bias.unsqueeze(0)], 0))
    else:
      self.wb.write_mat(self.weight.t())


if __name__ == '__main__':
  class Net(torch.nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = PimLinear(5, 5, 1, PimArrayType.SimpleLogicArray)

    def forward(self, x):
      output = self.fc1(x)
      return output


  model = Net()
  input = torch.randn([1, 5])
  output = model.forward(input)
  output.sum().backward()

  torch.nn.Parameter
