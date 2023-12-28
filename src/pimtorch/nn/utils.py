import torch
import torch.nn as nn
from src.pimtorch.config.globalCfg import globalCfg as cfg
from src.pimtorch.nn.modules import Linear, Conv2d, ReLU, Dropout

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filename, patience=7, verbose=False, delta=0, score_type='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.filename = filename
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.val_acc_max = 0.0
        self.delta = delta
        self.score_type = score_type

    def __call__(self, score_indicator, model):
        if self.score_type == 'accuracy':
            score = score_indicator
        else:  # loss
            score = -score_indicator

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score_indicator, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score_indicator, model)
            self.counter = 0

    def save_checkpoint(self, score_indicator, model):
        """Saves model when score become better."""
        torch.save(model.state_dict(), self.filename)

        if self.score_type == 'accuracy':
            if self.verbose:
                print(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {score_indicator:.4f}). '
                      f'Saving model ...')
            self.val_acc_max = score_indicator
            pass
        else:  # loss
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {score_indicator:.4f}). '
                      f'Saving model ...')
            self.val_loss_min = score_indicator

def PIMWarpper(torch_nn, *args, **kwargs):
  if isinstance(torch_nn, nn.Linear):
    return Linear(in_features=torch_nn.in_features,
                  out_features=torch_nn.out_features,
                  bias=torch_nn.bias is not None,
                  output_bit_width=cfg.outputBitWidth,
                  weight_bit_width=cfg.weightBitWidth,
                  grad_output_bit_width=cfg.gradOutputBitWidth,
                  compute_weight_bit_width=cfg.computeWeightBitWidth,
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
      output_bit_width=cfg.outputBitWidth,
      weight_bit_width=cfg.weightBitWidth,
      grad_output_bit_width=cfg.gradOutputBitWidth,
      compute_weight_bit_width=cfg.computeWeightBitWidth,
      *args, **kwargs)
  elif isinstance(torch_nn, nn.ReLU):
    return ReLU()
  elif isinstance(torch_nn, nn.Dropout):
    return Dropout(p = torch_nn.p)
  else:
    raise NotImplementedError("This type not supported: %s." % torch_nn.__class__)