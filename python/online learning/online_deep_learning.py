# -*- coding: utf-8 -*-
import numpy as np
import torch
import visdom
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from progressbar import *
from onn.OnlineNeuralNetwork import ONN
from torch.utils.data.dataset import Dataset

class SusyDataset(Dataset):
  def __init__(self, npz_file):
    npz = np.load(npz_file)
    self.x = npz['x_train']
    self.y = npz['y_train']

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]).argmax()


def train_onn(onn_network, device, train_loader, test_loader, test_interval, tb_writer):
  onn_network.train()
  widgets = ['Training: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
  bar = ProgressBar(widgets=widgets, maxval=len(train_loader))
  bar.start()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device).flatten(start_dim=1), target.to(device)
    onn_network.partial_fit(data, target, batch_idx, True)
    if batch_idx % test_interval == 0:
      test_onn(onn_network, device, test_loader, batch_idx, tb_writer)
    bar.update(batch_idx + 1)
  bar.finish()

def test_onn(onn_network, device, test_loader, current_step, vis):
  onn_network.eval()
  correct = 0
  widgets = ['Testing: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
  bar = ProgressBar(widgets=widgets, maxval=len(test_loader))
  bar.start()
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device).flatten(start_dim=1), target.to(device)
      predictions = onn_network.predict(data)
      correct += predictions.eq(target.view_as(predictions)).sum().item()
      bar.update(batch_idx+1)
  bar.finish()
  accuracy = 100. * correct / len(test_loader.dataset)
  # print('\nTest set: Accuracy: {}/{} ({:.6f}%)\n'.format(
  #   correct, len(test_loader.dataset), accuracy))
  # tb_writer.add_scalar('accuracy/test', accuracy, current_step)
  vis.line(X=torch.Tensor([current_step]), Y=torch.Tensor([accuracy]), win='Test Accuracy',
           update='append', opts={'title': 'Test Accuracy', 'xlabel': 'step', 'ylabel': 'accuracy'})


if __name__ == '__main__':
  features_size = 18
  batch_size = 10
  n_classes = 2
  test_interval = 10000
  use_cuda = False

  if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print("CUDA available! Training on GPU.")
  else:
    device = torch.device("cpu")
    print("Training on CPU.")

  # susy_dataset = SusyDataset("/home/zhouheng/datasets/susy.npz")
  susy_dataset = SusyDataset("/Users/zhouheng/Downloads/susy.npz")

  # tb_writer = SummaryWriter()
  vis_env = "Online Deep Learning"
  vis = visdom.Visdom(env=vis_env)
  vis.close(env=vis_env)

  n_val = int(len(susy_dataset) * 0.8)
  n_train = len(susy_dataset) - n_val
  train, val = torch.utils.data.random_split(susy_dataset, [n_train, n_val])
  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(val, batch_size=1000)

  # online learning
  onn_network = ONN(features_size=features_size, max_num_hidden_layers=8, qtd_neuron_per_hidden_layer=100,
                    n_classes=n_classes, batch_size=batch_size, b=0.99, n=0.01, s=0.2, use_cuda=use_cuda, vis=vis)
  onn_network.to(device, torch.double)
  train_onn(onn_network, device, train_loader, test_loader, test_interval, vis)

  vis.save(envs=[vis_env])
  # tb_writer.export_scalars_to_json("./all_scalars.json")
  # tb_writer.close()

