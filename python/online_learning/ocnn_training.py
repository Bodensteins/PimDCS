# -*- coding: utf-8 -*-
import numpy as np
import torch
import visdom
import yaml
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from progressbar import *
from onn.OnlineConvolutionNeuralNetwork import OCNN
from torch.utils.data.dataset import Dataset

class NumpyDataset(Dataset):
  def __init__(self, npz_file):
    npz = np.load(npz_file)
    self.x = torch.from_numpy(npz['x_train'])
    self.y = torch.from_numpy(npz['y_train'])

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx].argmax()


def train_onn(onn_network, device, train_loader, epoch):
  onn_network.train()
  widgets = ['[%d]Training: ' % epoch, Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
  bar = ProgressBar(widgets=widgets, maxval=len(train_loader))
  bar.start()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device=device, dtype=torch.float64), target.to(device)
    onn_network.partial_fit(data, target, epoch * len(train_loader) + batch_idx, True)
    bar.update(batch_idx + 1)
  bar.finish()

def test_onn(onn_network, device, test_loader, vis=None):
  onn_network.eval()
  correct = 0
  widgets = ['Testing: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
  bar = ProgressBar(widgets=widgets, maxval=len(test_loader))
  bar.start()
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
      data, target = data.to(device=device, dtype=torch.float64), target.to(device)
      predictions = onn_network.predict(data)
      correct += predictions.eq(target.view_as(predictions)).sum().item()
      bar.update(batch_idx+1)
  bar.finish()
  accuracy = 100. * correct / len(test_loader.dataset)
  print("Test accuracy: %.4f" % accuracy)
  if vis is not None:
    vis.line(X=torch.Tensor([0]), Y=torch.Tensor([accuracy]), win='Test Accuracy',
             update='append', opts={'title': 'Test Accuracy', 'xlabel': 'step', 'ylabel': 'accuracy'})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', metavar='DIR',
                      help='path to dataset')
  parser.add_argument('--dataset-name', metavar='NAME',
                      help='name of dataset')
  parser.add_argument('--info', metavar='STR',
                      help='detail info of this training')
  parser.add_argument('--network-config', metavar='PATH',
                      help='file path to network config')
  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                      help='number of data loading workers (default: 4)')
  parser.add_argument('--batch-size', default=1, type=int,
                      metavar='N', help='mini-batch size (default: 1)')
  parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                      help='training epochs (default: 1)')
  parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                      metavar='LR', help='initial learning rate')
  parser.add_argument('--beta', default=0.99, type=float,
                      help='discount rate parameter')
  parser.add_argument('--s', default=0.2, type=float,
                      help='s encourages all classifiers at every depth to affect the backprop update')
  parser.add_argument('--freeze-threshold', default=0.05, type=float,
                      help='freeze threshold')
  parser.add_argument('--classes', default=2, type=int,
                      help='number of categories')
  parser.add_argument('--log-interval', default=1000, type=int,
                      help='log interval')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--no-tfboard', action='store_true', default=False,
                      help='disables tensorboard')
  args = parser.parse_args()

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  if not use_cuda:
    torch.set_num_threads(args.workers)
  device = torch.device("cuda" if use_cuda else "cpu")

  if args.dataset_name == "mnist":
    train_dataset = MNIST(root=args.data, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root=args.data, train=False, transform=transforms.ToTensor())
  elif args.dataset_name == "cifar10":
    train_dataset = CIFAR10(root=args.data, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = CIFAR10(root=args.data, train=False, transform=transforms.ToTensor())
  else:
    print("This dataset is not supported.")
    exit(0)

  vis_env = "%s_%s" % (args.dataset_name, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
  vis = visdom.Visdom(env=vis_env)
  vis.close(env=vis_env)

  tb_writer = None if args.no_tfboard else SummaryWriter()

  text_win = vis.text("Training arguments:")
  for arg_name in vars(args):
    vis.text("%s: %s" % (arg_name, getattr(args, arg_name)), win=text_win, append=True)
  text_win = vis.text("freezed steps:", win=text_win, append=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=args.workers)

  with open(args.network_config, "r") as f:
    cfgs = yaml.full_load(f)

  # online_learning
  onn_network = OCNN(vgg_cfgs=cfgs['vgg_cfgs'], n_classes=args.classes, batch_size=args.batch_size,
                     b=args.beta, n=args.lr, s=args.s, freeze_threshold=args.freeze_threshold,
                     log_interval=args.log_interval, use_cuda=use_cuda, vis=vis, tb_writer=tb_writer)
  onn_network.to(device, torch.double)

  for epoch in range(args.epochs):
    train_onn(onn_network, device, train_loader, epoch)
  test_onn(onn_network, device, test_loader, None)

  for i, steps in enumerate(onn_network.freeze_steps):
    vis.text("freeze steps of layer %d: %d (%f%%)" % (i, steps, steps/len(train_loader)), win=text_win, append=True)
  vis.text("Total steps: %d" % len(train_loader), win=text_win, append=True)
  vis.save(envs=[vis_env])

  tb_writer.add_scal


