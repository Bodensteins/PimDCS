import collections
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from mab import algs


# cfgs = [
#   {"in_channels": 1, "block_cfg": [64, 'M'], "output_size": 12544},
#   {"in_channels": 64, "block_cfg": [128, 'M'], "output_size": 6272},
#   {"in_channels": 128, "block_cfg": [256, 256, 'M'], "output_size": 2304},
# ]

# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

def make_vgg_block(in_channels, cfg):
  layers = []
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)

def make_classifier(in_size, n_classes):
  return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

class OCNN(nn.Module):
  def __init__(self, vgg_cfgs, n_classes, batch_size=1,
               b=0.99, n=0.01, s=0.2, freeze_threshold=0.005, log_interval=1000, use_cuda=False, vis=None, tb_writer=None):
    super(OCNN, self).__init__()

    self.device = torch.device(
      "cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    self.vgg_blocks_num = len(vgg_cfgs)
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.b = Parameter(torch.tensor(
      b), requires_grad=False).to(self.device)
    self.n = Parameter(torch.tensor(
      n), requires_grad=False).to(self.device)
    self.s = Parameter(torch.tensor(
      s), requires_grad=False).to(self.device)

    self.vgg_blocks = []
    self.classifiers = []

    for cfg in vgg_cfgs:
      self.vgg_blocks.append(make_vgg_block(cfg["in_channels"], cfg["block_cfg"]))
      self.classifiers.append(make_classifier(cfg["output_size"], self.n_classes))

    # self.hidden_layers.append(
    #   nn.Linear(features_size, qtd_neuron_per_hidden_layer))
    #
    # for i in range(max_num_hidden_layers - 1):
    #   self.hidden_layers.append(
    #     nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
    #
    # for i in range(max_num_hidden_layers):
    #   self.output_layers.append(
    #     nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

    self.vgg_blocks = nn.ModuleList(self.vgg_blocks).to(self.device)
    self.classifiers = nn.ModuleList(self.classifiers).to(self.device)

    self.alpha = Parameter(torch.Tensor(self.vgg_blocks_num).fill_(1 / (self.vgg_blocks_num + 1)),
                           requires_grad=False).to(self.device)

    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.freeze_threshold = self.s / self.vgg_blocks_num + freeze_threshold
    self.freeze_steps = [0] * self.vgg_blocks_num

    self.cumulative_error = 0
    self.log_interval = log_interval
    self.vis = vis
    self.tb_writer = tb_writer

  def zero_grad(self):
    for i in range(self.vgg_blocks_num):
      self.classifiers[i].zero_grad()
      self.vgg_blocks[i].zero_grad()
      # self.output_layers[i].weight.grad.data.fill_(0)
      # self.output_layers[i].bias.grad.data.fill_(0)
      # self.hidden_layers[i].weight.grad.data.fill_(0)
      # self.hidden_layers[i].bias.grad.data.fill_(0)

  def update_weights(self, X, Y, batch_idx, show_loss):

    predictions_per_layer = self.forward(X)

    losses_per_layer = []

    for out in predictions_per_layer:
      loss = self.criterion(out, Y)
      losses_per_layer.append(loss)

    # w = [None] * len(losses_per_layer)
    # b = [None] * len(losses_per_layer)
    # grad_list = [ [grad... * sublayer] * losses_per_layer ]
    grad_list = [None] * len(losses_per_layer)
    mean_delta_w = {}
    mean_delta_b = {}

    with torch.no_grad():
      for i in range(len(losses_per_layer)):
        losses_per_layer[i].backward(retain_graph=True)
        for name, param in self.classifiers[i].named_parameters():
          delta = self.n * self.alpha[i] * param.grad.data
          param.data -= delta
          if "weight" in name:
            mean_delta_w["classifiers_%d.%s" % (i, name)] = delta
          else:
            mean_delta_b["classifiers_%d.%s" % (i, name)] = delta

        for j in range(i + 1):
          if grad_list[j] is None:
            grad_list[j] = []
            for param in self.vgg_blocks[j].parameters():
              grad_list[j].append(self.alpha[i] * param.grad.data)
          else:
            for idx, param in enumerate(self.vgg_blocks[j].parameters()):
              grad_list[j][idx] += self.alpha[i] * param.grad.data

        self.zero_grad()

      for i in range(len(losses_per_layer)):
        if self.alpha[i].item() >= self.freeze_threshold or batch_idx % 2 == 0:
          for param, grad in zip(self.vgg_blocks[i].parameters(), grad_list[i]):
            param.data -= self.n * grad

        else:
          self.freeze_steps[i] += 1

      for i in range(len(losses_per_layer)):
        self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
        self.alpha[i] = torch.max(self.alpha[i], self.s / self.vgg_blocks_num)

    z_t = torch.sum(self.alpha)
    self.alpha.data = self.alpha.data / z_t

    real_output = torch.sum(torch.mul(
      self.alpha.view(self.vgg_blocks_num, 1).repeat(1, self.batch_size).view(
        self.vgg_blocks_num, self.batch_size, 1), predictions_per_layer), 0)
    self.cumulative_error += torch.argmax(real_output, dim=1).ne(Y).sum().item()

    if show_loss and batch_idx % self.log_interval == 0:
      loss = self.criterion(real_output, Y)
      if self.vis is not None:
        self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([loss]), win='Final Loss of Train',
                      update='append', opts={'title': 'Final Loss of Train', 'xlabel': 'step', 'ylabel': 'loss'})
        self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([self.cumulative_error / (batch_idx + 1) * self.batch_size]),
                      win='Cumulative Error Rate', update='append',
                      opts={'title': 'Cumulative Error Rate', 'xlabel': 'step', 'ylabel': 'error rate'})
        for i in range(len(losses_per_layer)):
          self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([losses_per_layer[i]]), win='Train Loss',
                        name='layer %d' % i, update='append',
                        opts={'title': 'Train Loss',
                              'xlabel': 'step',
                              'ylabel': 'loss',
                              'showlegend': True})
          self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([self.alpha[i]]), win='Alpha', name='layer %d' % i,
                        update='append', opts={'title': 'Alpha',
                                               'xlabel': 'step',
                                               'ylabel': 'alpha',
                                               'showlegend': True})
      if self.tb_writer is not None:
        for i in range(len(losses_per_layer)):
          for name, param in self.classifiers[i].named_parameters():
            self.tb_writer.add_histogram("classifiers_%d/" % i + name, param.data, batch_idx)
          for name, param in self.vgg_blocks[j].named_parameters():
            self.tb_writer.add_histogram("vgg_blocks_%d/" % i + name, param.data, batch_idx)

  def forward(self, X):
    hidden_connections = []

    x = self.vgg_blocks[0](X)
    hidden_connections.append(x)

    for i in range(1, self.vgg_blocks_num):
      hidden_connections.append(self.vgg_blocks[i](hidden_connections[i - 1]))

    output_class = []

    for i in range(self.vgg_blocks_num):
      output_class.append(self.classifiers[i](hidden_connections[i]))

    pred_per_layer = torch.stack(output_class)

    return pred_per_layer

  def validate_input_X(self, data):
    if len(data.shape) != 2:
      raise Exception(
        "Wrong dimension for this X data. It should have only two dimensions.")

  def validate_input_Y(self, data):
    if len(data.shape) != 1:
      raise Exception(
        "Wrong dimension for this Y data. It should have only one dimensions.")

  def partial_fit_(self, X_data, Y_data, batch_idx, show_loss=True):
    # self.validate_input_X(X_data)
    # self.validate_input_Y(Y_data)
    self.update_weights(X_data, Y_data, batch_idx, show_loss)

  def partial_fit(self, X_data, Y_data, batch_idx, show_loss=True):
    self.partial_fit_(X_data, Y_data, batch_idx, show_loss)

  def predict_(self, X_data):
    # self.validate_input_X(X_data)
    return torch.argmax(torch.sum(torch.mul(
      self.alpha.view(self.vgg_blocks_num, 1).repeat(1, len(X_data)).view(
        self.vgg_blocks_num, len(X_data), 1), self.forward(X_data)), 0), dim=1)

  def predict(self, X_data):
    pred = self.predict_(X_data)
    return pred

  def export_params_to_json(self):
    state_dict = self.state_dict()
    params_gp = {}
    for key, tensor in state_dict.items():
      params_gp[key] = tensor.cpu().numpy().tolist()

    return json.dumps(params_gp)

  def load_params_from_json(self, json_data):
    params = json.loads(json_data)
    o_dict = collections.OrderedDict()
    for key, tensor in params.items():
      o_dict[key] = torch.tensor(tensor).to(self.device)
    self.load_state_dict(o_dict)


class OCNN_THS(OCNN):
  def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, b=0.99, n=0.01,
               s=0.2, e=[0.5, 0.35, 0.2, 0.1, 0.05], use_cuda=False):
    super().__init__(features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, b=b, n=n, s=s,
                     use_cuda=use_cuda)
    self.e = Parameter(torch.tensor(e), requires_grad=False)
    self.arms_values = Parameter(
      torch.arange(n_classes), requires_grad=False)
    self.explorations_mab = []

    for i in range(n_classes):
      self.explorations_mab.append(algs.ThompsomSampling(len(e)))

  def partial_fit(self, X_data, Y_data, exp_factor, batch_idx, show_loss=True):
    self.partial_fit_(X_data, Y_data, batch_idx, show_loss)
    self.explorations_mab[Y_data[0]].reward(exp_factor)

  def predict(self, X_data):
    pred = self.predict_(X_data)[0]
    exp_factor = self.explorations_mab[pred].select()[0]
    if np.random.uniform() < self.e[exp_factor]:
      removed_arms = self.arms_values.clone().numpy().tolist()
      removed_arms.remove(pred)
      return random.choice(removed_arms), exp_factor

    return pred, exp_factor
