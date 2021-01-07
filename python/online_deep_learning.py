# -*- coding: utf-8 -*-
import numpy as np
import torch
import visdom
from onn.OnlineNeuralNetwork import ONN
from torch.utils.data.dataset import Dataset

vis_env = "Online Deep Learning"
vis = visdom.Visdom(env=vis_env)
vis.close(env=vis_env)

class SusyDataset(Dataset):
  def __init__(self, npz_file):
    npz = np.load(npz_file)
    self.x = npz['x_train']
    self.y = npz['y_train']

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]).argmax()


def train_onn(onn_network, device, train_loader, test_loader, test_interval):
  onn_network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device).flatten(start_dim=1), target.to(device)
    onn_network.partial_fit(batch_idx, data, target)
    if batch_idx % test_interval == 0:
      test_onn(onn_network, device, test_loader, batch_idx)


def test_onn(onn_network, device, test_loader, current_step):
  onn_network.eval()
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device).flatten(start_dim=1), target.to(device)
      predictions = onn_network.predict(data)
      correct += predictions.eq(target.view_as(predictions)).sum().item()
  accuracy = 100. * correct / len(test_loader.dataset)
  print('\nTest set: Accuracy: {}/{} ({:.6f}%)\n'.format(
    correct, len(test_loader.dataset), accuracy))
  vis.line(X=torch.Tensor([current_step]), Y=torch.Tensor([accuracy]), win='Test Accuracy',
           update='append', opts={'title': 'Test Accuracy', 'xlabel': 'step', 'ylabel': 'accuracy'})


if __name__ == '__main__':
  features_size = 18
  batch_size = 10
  n_classes = 2
  test_interval = 50000

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  susy_dataset = SusyDataset("/Users/zhouheng/Downloads/susy.npz")

  n_val = int(len(susy_dataset) * 0.8)
  n_train = len(susy_dataset) - n_val
  train, val = torch.utils.data.random_split(susy_dataset, [n_train, n_val])
  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(val, batch_size=1000)

  # Online Learning
  onn_network = ONN(features_size=features_size, max_num_hidden_layers=3, qtd_neuron_per_hidden_layer=100,
                    n_classes=n_classes, batch_size=batch_size, b=0.99, n=0.01, s=0.2, visdom=vis)
  onn_network.to(device, torch.double)
  train_onn(onn_network, device, train_loader, test_loader, test_interval)

