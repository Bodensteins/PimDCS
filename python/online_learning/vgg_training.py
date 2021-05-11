# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.vgg import cfgs, make_layers
from torch.optim.lr_scheduler import StepLR


class VGG(nn.Module):
  def __init__(self, features, num_classes=1000, init_weights=True):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, num_classes),
    )
    if init_weights:
      self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def train(args, model, device, train_loader, optimizer, epoch, vis):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            vis.line(X=torch.Tensor([(epoch-1)*len(train_loader.dataset)+batch_idx * len(data)]),
                     Y=torch.Tensor([loss]), win='Loss of Train',
                     update='append', opts={'title': 'Loss of Train', 'xlabel': 'step', 'ylabel': 'loss'})
            if args.dry_run:
                break


def test(model, device, test_loader, epoch, vis):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    vis.line(X=torch.Tensor([epoch]),
             Y=torch.Tensor([100. * correct / len(test_loader.dataset)]), win='Test Accuracy',
             update='append', opts={'title': 'Test Accuracy', 'xlabel': 'epoch', 'ylabel': 'Accuracy'})

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset-name', metavar='NAME',
                        help='name of dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    vis_env = "vgg_%s" % args.dataset_name
    vis = visdom.Visdom(env=vis_env)
    vis.close(env=vis_env)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
      torch.set_num_threads(args.workers)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10(root=args.data, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = VGG(make_layers(cfgs['A'], batch_norm=False), num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, vis)
        test(model, device, test_loader, epoch, vis)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")


if __name__ == '__main__':
    main()


