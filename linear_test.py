# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from numpy.testing import assert_almost_equal
from numpy_ml.neural_nets.layers import Conv2D
from numpy_ml.utils.testing import random_tensor


def torchify(var, requires_grad=True):
  return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)

def err_fmt(params, golds, ix, warn_str=""):
  mine, label = params[ix]
  err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
  prev_mine, prev_label = params[max(ix - 1, 0)]
  err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
    prev_label, prev_mine, prev_label, golds[prev_label]
  )
  err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
    label, mine, label, golds[label]
  )
  err_msg += warn_str
  err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
  return err_msg


def remove_padding(x, padding):
  out_shape = [x.size(0), x.size(1), x.size(2) - padding[2]-padding[3], x.size(3)-padding[0]-padding[1]]
  mask = torch.ones(out_shape)
  mask = torch.nn.functional.pad(mask, padding, "constant", 0)
  return x[mask.bool()].view(out_shape)
  # return x[:, :, padding[2]:-padding[3], padding[0]:-padding[1]]


def insert_zeros(x, stride):
  if stride == 1:
    return x
  else:
    w = x.new_zeros(stride, stride)
    w[0, 0] = 1
    res = torch.nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))
    return remove_padding(res, [0, stride-1, 0, stride-1])
    # return torch.nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))[:,:,:-(stride-1),:-(stride-1)]



def my_backward(grads, N, Cin, Hin, Win, Cout, kH, kW, H_, W_, stride, padding, dilation):
    X = grads["X"]
    W = grads["W"]
    b = grads["b"]
    dLdZ = grads["dLdZ"]

    # gold
    dLdW = grads["dLdW"]
    dLdB = grads["dLdB"]
    dLdX = grads["dLdX"]

    # my backward
    flip_W = W.flip((2, 3))
    swap_flip_W = flip_W.permute(1, 0, 2, 3)
    insert_dLdZ = insert_zeros(dLdZ, stride)

    unf_dLdZ = torch.nn.functional.unfold(insert_dLdZ, (kH, kW), dilation=1, padding=kW-1, stride=1)
    kernel = swap_flip_W.reshape(swap_flip_W.size(0), -1).t()
    dZ_ = unf_dLdZ.transpose(1, 2).matmul(swap_flip_W.reshape(swap_flip_W.size(0), -1).t()).transpose(1, 2)
    dZ = torch.nn.functional.fold(dZ_, output_size=(Hin, Win), kernel_size=(1, 1), padding=padding)

    if not torch.allclose(dLdX, dZ):
      print((dZ - dLdX).abs().max())

    flip_X = X.flip((2, 3))
    swap_flip_X = flip_X.permute(1, 0, 2, 3)

    unf_swap_dLdZ = torch.nn.functional.unfold(insert_dLdZ.permute(1, 0, 2, 3), (Hin, Win), dilation=1, padding=kW-1, stride=1)
    kernel = swap_flip_X.reshape(swap_flip_X.size(0), -1).t()
    dW_ = unf_swap_dLdZ.transpose(1, 2).matmul(swap_flip_X.reshape(swap_flip_X.size(0), -1).t()).transpose(1, 2)
    dW = torch.nn.functional.fold(dW_, output_size=(kH, kW), kernel_size=(1, 1), padding=padding)

    db = dLdZ.sum(dim=[0, 2, 3])
    return dLdZ

class TorchConv2DLayer(nn.Module):
  def __init__(self, in_channels, out_channels, act_fn, params, hparams, **kwargs):
    super(TorchConv2DLayer, self).__init__()

    W = params["W"]
    b = params["b"]
    self.act_fn = act_fn

    self.layer1 = nn.Conv2d(
      in_channels,
      out_channels,
      hparams["kernel_shape"],
      padding=hparams["pad"],
      stride=hparams["stride"],
      dilation=hparams["dilation"] + 1,
      bias=True,
    )

    # (f[0], f[1], n_in, n_out) -> (n_out, n_in, f[0], f[1])
    W = np.moveaxis(W, [0, 1, 2, 3], [-2, -1, -3, -4])
    assert self.layer1.weight.shape == W.shape
    assert self.layer1.bias.shape == b.flatten().shape

    self.layer1.weight = nn.Parameter(torch.FloatTensor(W))
    self.layer1.bias = nn.Parameter(torch.FloatTensor(b.flatten()))

  def forward(self, X):
    # (N, H, W, C) -> (N, C, H, W)
    self.X = np.moveaxis(X, [0, 1, 2, 3], [0, -2, -1, -3])
    if not isinstance(self.X, torch.Tensor):
      self.X = torchify(self.X)

    self.X.retain_grad()

    self.Z = self.layer1(self.X)
    self.Z.retain_grad()

    self.Y = self.act_fn(self.Z)
    self.Y.retain_grad()
    return self.Y

  def extract_grads(self, X):
    self.forward(X)
    self.loss = self.Y.sum()
    self.loss.backward()

    # W (theirs): (n_out, n_in, f[0], f[1]) -> W (mine): (f[0], f[1], n_in, n_out)
    # X (theirs): (N, C, H, W)              -> X (mine): (N, H, W, C)
    # Y (theirs): (N, C, H, W)              -> Y (mine): (N, H, W, C)
    orig, X_swap, W_swap = [0, 1, 2, 3], [0, -1, -3, -2], [-1, -2, -4, -3]
    grads = {
      "X": np.moveaxis(self.X.detach().numpy(), orig, X_swap),
      "W": np.moveaxis(self.layer1.weight.detach().numpy(), orig, W_swap),
      "b": self.layer1.bias.detach().numpy().reshape(1, 1, 1, -1),
      "y": np.moveaxis(self.Y.detach().numpy(), orig, X_swap),
      "dLdY": np.moveaxis(self.Y.grad.numpy(), orig, X_swap),
      "dLdZ": np.moveaxis(self.Z.grad.numpy(), orig, X_swap),
      "dLdW": np.moveaxis(self.layer1.weight.grad.numpy(), orig, W_swap),
      "dLdB": self.layer1.bias.grad.numpy().reshape(1, 1, 1, -1),
      "dLdX": np.moveaxis(self.X.grad.numpy(), orig, X_swap),
    }

    torch_grads = {
      "X": self.X.detach(),
      "W": self.layer1.weight.detach(),
      "b": self.layer1.bias.detach(),
      "y": self.Y.detach(),
      "dLdY": self.Y.grad,
      "dLdZ": self.Z.grad,
      "dLdW": self.layer1.weight.grad,
      "dLdB": self.layer1.bias.grad,
      "dLdX": self.X.grad,
    }

    return grads, torch_grads

class TorchLinearActivation(nn.Module):
  def __init__(self):
    super(TorchLinearActivation, self).__init__()
    pass

  @staticmethod
  def forward(input):
    return input

  @staticmethod
  def backward(grad_output):
    return torch.ones_like(grad_output)

def test_Conv2D(N=15):
  from numpy_ml.neural_nets.layers import Conv2D
  from numpy_ml.neural_nets.activations import Tanh, ReLU, Sigmoid, Affine

  N = np.inf if N is None else N

  np.random.seed(12345)

  acts = [
    (Tanh(), nn.Tanh(), "Tanh"),
    (Sigmoid(), nn.Sigmoid(), "Sigmoid"),
    (ReLU(), nn.ReLU(), "ReLU"),
    (Affine(), TorchLinearActivation(), "Affine"),
  ]

  i = 1
  while i < N + 1:
    # choose arguments randomly

    # n_ex = np.random.randint(1, 10)
    # in_rows = np.random.randint(1, 10)
    # in_cols = np.random.randint(1, 10)
    # n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)
    # f_shape = (
    #   min(in_rows, np.random.randint(1, 5)),
    #   min(in_cols, np.random.randint(1, 5)),
    # )
    # p, s = np.random.randint(0, 5), np.random.randint(1, 3)
    # d = np.random.randint(0, 5)

    # n_ex = 3
    # in_rows = 6
    # in_cols = 2
    # n_in, n_out = 1, 2
    # f_shape = (3, 2)
    # p, s = 2, 2
    # d = 1

    n_ex = 1
    in_rows = 4
    in_cols = 4
    n_in, n_out = 2, 3
    f_shape = (3, 3)
    p, s = 0, 1
    d = 0       # numpy-ml default is 0, but pytorch is 1.

    fr, fc = f_shape[0] * (d + 1) - d, f_shape[1] * (d + 1) - d
    out_rows = int(1 + (in_rows + 2 * p - fr) / s)
    out_cols = int(1 + (in_cols + 2 * p - fc) / s)

    if out_rows <= 0 or out_cols <= 0:
      continue

    # X = random_tensor((n_ex, in_rows, in_cols, n_in), standardize=False)   # (N, H, W, C)
    X = np.arange(n_ex*n_in*in_rows*in_cols).reshape(n_ex, in_rows, in_cols, n_in)

    # randomly select an activation function
    act_fn, torch_fn, act_fn_name = acts[np.random.randint(0, len(acts))]

    # initialize Conv2D layer
    L1 = Conv2D(
      out_ch=n_out,
      kernel_shape=f_shape,
      act_fn=act_fn,
      pad=p,
      stride=s,
      dilation=d,
    )

    # forward prop
    y_pred = L1.forward(X)

    # backprop
    dLdy = np.ones_like(y_pred)
    dLdX = L1.backward(dLdy)

    # get gold standard gradients
    gold_mod = TorchConv2DLayer(
      n_in, n_out, torch_fn, L1.parameters, L1.hyperparameters
    )
    golds, torch_grads = gold_mod.extract_grads(X)

    my_backward(torch_grads, n_ex, n_in, in_rows, in_cols, n_out, f_shape[0], f_shape[1], out_rows, out_cols, s, p, d)

    params = [
      (L1.X[0], "X"),
      (y_pred, "y"),
      (L1.parameters["W"], "W"),
      (L1.parameters["b"], "b"),
      (L1.gradients["W"], "dLdW"),
      (L1.gradients["b"], "dLdB"),
      (dLdX, "dLdX"),
    ]

    print("\nTrial {}".format(i))
    print("pad={}, stride={}, f_shape={}, n_ex={}".format(p, s, f_shape, n_ex))
    print("in_rows={}, in_cols={}, n_in={}".format(in_rows, in_cols, n_in))
    print("out_rows={}, out_cols={}, n_out={}".format(out_rows, out_cols, n_out))
    print("dilation={}".format(d))
    for ix, (mine, label) in enumerate(params):
      assert_almost_equal(
        mine, golds[label], err_msg=err_fmt(params, golds, ix), decimal=-10
      )
      print("\tPASSED {}".format(label))
    i += 1

if __name__ == '__main__':
  test_Conv2D(N=1)