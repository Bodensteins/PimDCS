import torch.nn as nn
from ..nn.modules import Linear, Conv2d, ReLU, Dropout

def PIMWarpper(torch_nn, *args, **kwargs):
  if isinstance(torch_nn, nn.Linear):
    return Linear(in_features=torch_nn.in_features,
                       out_features=torch_nn.out_features,
                       bias=torch_nn.bias is not None,
                       *args, **kwargs)
  elif isinstance(torch_nn, nn.Conv2d):
    return Conv2d(
      in_channels=torch_nn.in_channels,
      out_channels=torch_nn.out_channels,
      kernel_size=torch_nn.kernel_size,
      stride=torch_nn.stride,
      padding=torch_nn.padding,
      dilation=torch_nn.dilation,
      groups=torch_nn.groups,
      bias=torch_nn.bias is not None,
      padding_mode=torch_nn.padding_mode,
      *args, **kwargs)
  elif isinstance(torch_nn, nn.ReLU):
    return ReLU(*args, **kwargs)
  elif isinstance(torch_nn, nn.Dropout):
    return Dropout(p = torch_nn.p, *args, **kwargs)
  else:
    raise TypeError("This type not supported: %s." % torch_nn.__class__)