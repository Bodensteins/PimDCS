from __future__ import print_function
import argparse
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pimtorch import PimLinear, PimArrayType, PimConv2d
from ftrl import FTRL
from onn.OnlineNeuralNetwork import ONN

class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        # original cnn
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        # pim linear
        # self.fc1 = PimLinear(784, 64, batch_size, PimArrayType.SimpleLogicArray)
        # self.fc2 = PimLinear(64, 32, batch_size, PimArrayType.SimpleLogicArray)
        # self.fc3 = PimLinear(32, 10, batch_size, PimArrayType.SimpleLogicArray)
        #
        # self.register_parameter('fc1_weight', torch.nn.Parameter(self.fc1.weight))
        # self.register_parameter('fc1_bias', torch.nn.Parameter(self.fc1.bias))
        # self.register_parameter('fc2_weight', torch.nn.Parameter(self.fc2.weight))
        # self.register_parameter('fc2_bias', torch.nn.Parameter(self.fc2.bias))
        # self.register_parameter('fc3_weight', torch.nn.Parameter(self.fc3.weight))
        # self.register_parameter('fc3_bias', torch.nn.Parameter(self.fc3.bias))

        # FTRL
        # self.conv1 = nn.Conv2d(1, 32, 3, 2)
        # self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        # original cnn
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output

        # pim linear
        # x = self.fc1.forward(x.view([-1, 784]))
        # x = F.relu(x)
        # x = self.fc2.forward(x)
        # x = F.relu(x)
        # x = self.fc3.forward(x)
        # output = F.log_softmax(x, dim=1)
        #
        # print(self.fc3.weight)
        # self.fc3.print_pim_weight()
        # return output

        # FTRL
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_onn(onn_network, device, train_loader):
    onn_network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).flatten(start_dim=1), target.to(device)
        onn_network.partial_fit(batch_idx, data, target)

def test_onn(onn_network, device, test_loader):
    onn_network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).flatten(start_dim=1), target.to(device)
            predictions = onn_network.predict(data)
            correct += predictions.eq(target.view_as(predictions)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    vis_env = "Online Deep Learning"
    vis = visdom.Visdom(env=vis_env)
    vis.close(env=vis_env)

    torch.manual_seed(args.seed)

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
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = Net(args.batch_size).to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = FTRL(model.parameters(), alpha=1.0, beta=1.0, l1=0.5, l2=0.5)

    # # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    #     # scheduler.step()
    #
    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


    # Online Learning
    onn_network = ONN(features_size=784, max_num_hidden_layers=3, qtd_neuron_per_hidden_layer=100, n_classes=10,
                      batch_size=args.batch_size, b=0.95, n=0.01, s=0.3, visdom=vis)
    train_onn(onn_network, device, train_loader)
    test_onn(onn_network, device, test_loader)

if __name__ == '__main__':
    main()
