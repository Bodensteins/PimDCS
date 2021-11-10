from __future__ import print_function
import argparse
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pimlinear as pl
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from quantization import de_quantization, int_to_float
import pim_optimizer as po

from quantization import *


class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
       
        x1 = torch.flatten(x, 1)
        x2 = self.fc1(x1)
        x3 = F.relu(x2)
        x4 = self.fc2(x3)
        output = F.log_softmax(x4, dim=1)
        return output, x2, x3, x4


class PimNet(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.fc1 = pl.PimLinear(784, 128, 8, 3, 8, device=device)
        self.fc2 = pl.PimLinear(128, 10, 8, 3, 8, device=device)
        self.relu = pl.PimRelu()
        self.dequan = pl.DeQuanLayer(8)

    def forward(self, x):
        x = torch.flatten(x, 1)

        # ox = x.clone().detach().requires_grad_()
        ox = None
        x_p = pl.creat_quantization_para(bit_width=16, tensor_type=pl.TensorType.Normal, device=self.device)
        x_p.requires_grad_()
        # print(x.abs().max())
        x = pl.quantization_tensor(x_p, x)

        # print((de_quantization([x, x_p])-ox).abs().max())
        x, x_p, ox = self.fc1(x, x_p, ox)
        # print((de_quantization([x, x_p])-ox).abs().max())
        x, x_p, ox = self.relu(x, x_p, ox)
        # print((de_quantization([x, x_p])-ox).abs().max())
        x, x_p, ox = self.fc2(x, x_p, ox)
        # print((de_quantization([x, x_p]).detach()-ox.detach()).abs().max())
        x = self.dequan(x, x_p)
        output = F.log_softmax(x, dim=1)
        #out = F.log_softmax(ox, dim = 1)
        # print((output.detach()-out.detach()).abs().max())
        return output

def train(args, model, device, train_loader, optimizer, epoch, model2, optimizer2):
    model.train()
    if model2 != None:
        model2.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output2 = None
        if model2 != None:
            data2 = data.clone().detach().requires_grad_()
            target2 = target.clone().detach()
            optimizer2.zero_grad()
            output2, x2, x3, x4 = model2(data2)
            output2.requires_grad_()
            loss2 = F.nll_loss(output2, target2)
            loss2.backward()
            #print(model2.fc2.weight.grad.t())
            optimizer2.step()


        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        #print(model.fc2.weight.grad)
        optimizer.step()

        # t2diff = model.fc2.weight.data.detach()- \
        #     de_quantization([model.fc2.wArr.detach(), model.fc2.wArrConfig.detach()])

        # t1diff = model.fc1.weight.data.detach()- \
        #     de_quantization([model.fc1.wArr.detach(), model.fc1.wArrConfig.detach()]) 
        # print(t2diff.abs().max())#/t2diff.size(0)/t2diff.size(1))
        # print(t1diff.abs().max())#/t1diff.size(0)/t1diff.size(1))


        # tmp = torch.cat((model2.fc1.weight.data.t().clone(), model2.fc1.bias.data.clone().reshape(1, 128)), 0)
        # print(tmp - model.fc1.weight)
        if batch_idx % args.log_interval == 0:
            if model2 != None:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss2.item()))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, model2):
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
    if model2 == None:
        return
    model = model2
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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pim', action='store_true', default=True,
                        help='For use pim')
    parser.add_argument('--both', action='store_true', default=False,
                        help='Both two model runing')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

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
    model2, optimizer2 = None, None
    if args.pim:
        model = PimNet(args.batch_size, device=device).to(device)
        if args.both:
            model2 = Net(args.batch_size).to(device)
            model2.fc1.weight.data = model.fc1.weight[0:-1].t().clone().detach()
            model2.fc1.bias.data = model.fc1.weight[-1].clone().detach()
            model2.fc2.weight.data = model.fc2.weight[0:-1].t().clone().detach()
            model2.fc2.bias.data = model.fc2.weight[-1].clone().detach()
            optimizer2 = optim.SGD(model2.parameters(), lr = args.lr)
        optimizer = po.PimSGD(model, lr=args.lr, momentum=0.9, run_mode=po.OptimMode.float_weight)
    else:
        model = Net(args.batch_size).to(device)
        #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), lr = args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, model2, optimizer2)
        test(model, device, test_loader, model2)
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
