import collections
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mab import algs


class ONN(nn.Module):
  def __init__(self, features_size, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, batch_size=1,
               b=0.99, n=0.01, s=0.2, freeze_threshold=0.005, log_interval=1000, use_cuda=False, vis=None, tb_writer=None):
    super(ONN, self).__init__()

    self.device = torch.device(
      "cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    self.features_size = features_size
    self.max_num_hidden_layers = max_num_hidden_layers
    self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.b = Parameter(torch.tensor(
      b), requires_grad=False).to(self.device)
    self.n = Parameter(torch.tensor(
      n), requires_grad=False).to(self.device)
    self.s = Parameter(torch.tensor(
      s), requires_grad=False).to(self.device)

    self.hidden_layers = []
    self.output_layers = []

    self.hidden_layers.append(
      nn.Linear(features_size, qtd_neuron_per_hidden_layer))

    for i in range(max_num_hidden_layers - 1):
      self.hidden_layers.append(
        nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))

    for i in range(max_num_hidden_layers):
      self.output_layers.append(
        nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

    self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
    self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

    self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers + 1)),
                           requires_grad=False).to(self.device)

    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.freeze_threshold = self.s / self.max_num_hidden_layers + freeze_threshold
    self.freeze_steps = [0] * self.max_num_hidden_layers

    self.cumulative_error = 0
    self.log_interval = log_interval
    self.vis = vis
    self.tb_writer = tb_writer

    self.hidden_w_bar = []
    self.hidden_b_bar = []
    self.out_w_bar = []
    self.out_b_bar = []
    self.hidden_delta_w = [0.0] * max_num_hidden_layers
    self.hidden_delta_b = [0.0] * max_num_hidden_layers
    self.out_delta_w = [0.0] * max_num_hidden_layers
    self.out_delta_b = [0.0] * max_num_hidden_layers

  def zero_grad(self):
    for i in range(self.max_num_hidden_layers):
      self.output_layers[i].zero_grad()
      self.hidden_layers[i].zero_grad()
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

    w = [None] * len(losses_per_layer)
    b = [None] * len(losses_per_layer)
    mean_delta_w = {}
    mean_delta_b = {}

    with torch.no_grad():
      for i in range(len(losses_per_layer)):
        losses_per_layer[i].backward(retain_graph=True)
        delta_w = self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
        delta_b = self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
        self.output_layers[i].weight.data -= delta_w
        self.output_layers[i].bias.data -= delta_b
        mean_delta_w["out_%d.delta_w" % i] = delta_w.abs().mean().item()
        mean_delta_b["out_%d.delta_b" % i] = delta_b.abs().mean().item()
        self.out_delta_w[i] += mean_delta_w["out_%d.delta_w" % i]
        self.out_delta_b[i] += mean_delta_b["out_%d.delta_b" % i]

        if self.tb_writer is not None and batch_idx % self.log_interval == 0:
          self.tb_writer.add_histogram("delta/output_layer_%d.weight" % i, delta_w, batch_idx)
          self.tb_writer.add_histogram("delta/output_layer_%d.bias" % i, delta_b, batch_idx)
          self.tb_writer.add_histogram("weight/output_layer_%d.weight" % i, self.output_layers[i].weight.data, batch_idx)
          self.tb_writer.add_histogram("bias/output_layer_%d.bias" % i, self.output_layers[i].bias.data, batch_idx)

        for j in range(i + 1):
          if w[j] is None:
            w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
            b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
          else:
            w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
            b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

        self.zero_grad()

      for i in range(len(losses_per_layer)):
        if self.alpha[i].item() >= self.freeze_threshold or batch_idx % 2 == 0:
          delta_w = self.n * w[i]
          delta_b = self.n * b[i]
          self.hidden_layers[i].weight.data -= delta_w
          self.hidden_layers[i].bias.data -= delta_b
          mean_delta_w["hidden_%d.delta_w" % i] = delta_w.abs().mean().item()
          mean_delta_b["hidden_%d.delta_b" % i] = delta_b.abs().mean().item()
          self.hidden_delta_w[i] += mean_delta_w["hidden_%d.delta_w" % i]
          self.hidden_delta_b[i] += mean_delta_b["hidden_%d.delta_b" % i]

          if self.tb_writer is not None and batch_idx % self.log_interval == 0:
            self.tb_writer.add_histogram("delta/hidden_layer_%d.weight" % i, delta_w, batch_idx)
            self.tb_writer.add_histogram("delta/hidden_layer_%d.bias" % i, delta_b, batch_idx)
            self.tb_writer.add_histogram("weight/hidden_layers_%d.weight" % i, self.hidden_layers[i].weight.data, batch_idx)
            self.tb_writer.add_histogram("bias/hidden_layers_%d.bias" % i, self.hidden_layers[i].bias.data, batch_idx)
        else:
          self.freeze_steps[i] += 1

      for i in range(len(losses_per_layer)):
        self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
        self.alpha[i] = torch.max(self.alpha[i], self.s / self.max_num_hidden_layers)

    z_t = torch.sum(self.alpha)
    self.alpha.data = self.alpha.data / z_t

    real_output = torch.sum(torch.mul(
      self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, self.batch_size).view(
        self.max_num_hidden_layers, self.batch_size, 1), predictions_per_layer), 0)
    self.cumulative_error += torch.argmax(real_output, dim=1).ne(Y).sum().item()

    if batch_idx % 5000 == 0 and batch_idx != 0:
      self.hidden_w_bar.append(self.hidden_delta_w)
      self.hidden_b_bar.append(self.hidden_delta_b)
      self.out_w_bar.append(self.out_delta_w)
      self.out_b_bar.append(self.out_delta_b)
      self.hidden_delta_w = [0.0] * len(losses_per_layer)
      self.hidden_delta_b = [0.0] * len(losses_per_layer)
      self.out_delta_w = [0.0] * len(losses_per_layer)
      self.out_delta_b = [0.0] * len(losses_per_layer)
      if self.vis is not None:
        self.vis.bar(X=np.array(self.hidden_w_bar),
                     win='hidden weight',
                     opts={
                       "stacked": True,
                       'title': 'hidden weight',
                       "legend": ['layer %d' % i for i in range(len(losses_per_layer))],
                       # "rownames": ['step %d' % i * 10000 for i in range(batch_idx // 10000)]
                     })
        self.vis.bar(X=np.array(self.hidden_b_bar),
                     win='hidden bias',
                     opts={
                       "stacked": True,
                       'title': 'hidden bias',
                       "legend": ['layer %d' % i for i in range(len(losses_per_layer))],
                       # "rownames": ['step %d' % i * 10000 for i in range(batch_idx // 10000)]
                     })
        self.vis.bar(X=np.array(self.out_w_bar),
                     win='out weight',
                     opts={
                       "stacked": True,
                       'title': 'out weight',
                       "legend": ['layer %d' % i for i in range(len(losses_per_layer))],
                       # "rownames": ['step %d' % i * 10000 for i in range(batch_idx // 10000)]
                     })
        self.vis.bar(X=np.array(self.out_b_bar),
                     win='out bias',
                     opts={
                       "stacked": True,
                       'title': 'out bias',
                       "legend": ['layer %d' % i for i in range(len(losses_per_layer))],
                       # "rownames": ['step %d' % i * 10000 for i in range(batch_idx // 10000)]
                     })


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
          # self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([mean_delta_w["hidden_%d.delta_w" % i]]),
          #               win='hidden delta_w', name='layer %d' % i,
          #               update='append', opts={'title': 'hidden delta_w',
          #                                      'xlabel': 'step',
          #                                      'ylabel': 'delta',
          #                                      'showlegend': True})
          # self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([mean_delta_b["hidden_%d.delta_b" % i]]),
          #               win='hidden delta_b', name='layer %d' % i,
          #               update='append', opts={'title': 'hidden delta_b',
          #                                      'xlabel': 'step',
          #                                      'ylabel': 'delta',
          #                                      'showlegend': True})
          # self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([mean_delta_w["out_%d.delta_w" % i]]),
          #               win='out delta_w', name='layer %d' % i,
          #               update='append', opts={'title': 'out delta_w',
          #                                      'xlabel': 'step',
          #                                      'ylabel': 'delta',
          #                                      'showlegend': True})
          # self.vis.line(X=torch.Tensor([batch_idx]), Y=torch.Tensor([mean_delta_b["out_%d.delta_b" % i]]),
          #               win='out delta_b', name='layer %d' % i,
          #               update='append', opts={'title': 'out delta_b',
          #                                      'xlabel': 'step',
          #                                      'ylabel': 'delta',
          #                                      'showlegend': True})




  def forward(self, X):
    hidden_connections = []

    x = F.relu(self.hidden_layers[0](X))
    hidden_connections.append(x)

    for i in range(1, self.max_num_hidden_layers):
      hidden_connections.append(
        F.relu(self.hidden_layers[i](hidden_connections[i - 1])))

    output_class = []

    for i in range(self.max_num_hidden_layers):
      output_class.append(self.output_layers[i](hidden_connections[i]))

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
    self.validate_input_X(X_data)
    self.validate_input_Y(Y_data)
    self.update_weights(X_data, Y_data, batch_idx, show_loss)

  def partial_fit(self, X_data, Y_data, batch_idx, show_loss=True):
    self.partial_fit_(X_data, Y_data, batch_idx, show_loss)

  def predict_(self, X_data):
    self.validate_input_X(X_data)
    return torch.argmax(torch.sum(torch.mul(
      self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, len(X_data)).view(
        self.max_num_hidden_layers, len(X_data), 1), self.forward(X_data)), 0), dim=1)

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


class ONN_THS(ONN):
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
