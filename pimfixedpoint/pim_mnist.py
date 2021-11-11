from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pimlinear as pl
from torchvision import datasets, transforms
import pimconv as pc
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


class PimConvNet(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.conv1 = pc.PimConv2D((1, 28, 28), (5, 5), 10, device=device)
        self.conv2 = pc.PimConv2D((10, 12, 12), (5, 5), 20, device=device) 
        self.fc = pl.PimLinear(4*4*20, 10, device=device)
        self.dequan = pl.DeQuanLayer(16)

    def forward(self, x):

        x, x_f = pl.quanFunction.apply(x, 16)
        _, _, x = self.conv1(x, x_f)

        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x, x_f = pl.quanFunction.apply(x, 16)

        _, _, x = self.conv2(x, x_f)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x = torch.flatten(x, 1)
        x, x_f = pl.quanFunction.apply(x, 16)

        x, x_f = self.fc(x, x_f)
        x = self.dequan(x, x_f)

        output = F.log_softmax(x, dim=1)

        return output


class PimNet(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.fc1 = pl.PimLinear(784, 128, 16, 16, 16, device=device)
        self.fc2 = pl.PimLinear(128, 10, 16, 16, 16, device=device)
        self.relu = pl.PimRelu()
        self.dequan = pl.DeQuanLayer(16)

    def forward(self, x):
        x = torch.flatten(x, 1)

        # ox = x.clone().detach().requires_grad_()
        x_p = pl.creat_quantization_para(bit_width=16, tensor_type=pl.TensorType.Normal, device=self.device)
        x_p.requires_grad_()
        x = pl.quantization_tensor(x_p, x)

        x, x_p = self.fc1(x, x_p)
        x, x_p = self.relu(x, x_p)
        x, x_p = self.fc2(x, x_p)
        x = self.dequan(x, x_p)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_log_interval = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss_value = loss.item()
        loss_log_interval += loss_value
        loss.backward()
        # print(model.fc2.weight.grad)
        optimizer.step()

        # t2diff = model.fc2.weight.data.detach()- \
        #     de_quantization([model.fc2.wArr.detach(), model.fc2.wArrConfig.detach()])

        # t1diff = model.fc1.weight.data.detach()- \
        #     de_quantization([model.fc1.wArr.detach(), model.fc1.wArrConfig.detach()]) 
        # print(t2diff.abs().max())#/t2diff.size(0)/t2diff.size(1))
        # print(t1diff.abs().max())#/t1diff.size(0)/t1diff.size(1))

        # tmp = torch.cat((model2.fc1.weight.data.t().clone(), model2.fc1.bias.data.clone().reshape(1, 128)), 0)
        # print(tmp - model.fc1.weight)
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss_log_interval / args.log_interval))
            loss_log_interval = 0
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
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
    parser.add_argument('--pimconv', action='store_true', default=False,
                        help='For use pimconv')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:2" if use_cuda else "cpu")

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
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    if args.pim:
        if args.pimconv:
            model = PimConvNet(args.batch_size, device=device).to(device)
        else:
            model = PimNet(args.batch_size, device=device).to(device)

        optimizer = po.PimSGD(model, lr=args.lr, momentum=0.9, run_mode=po.OptimMode.full_fix)
    else:
        model = Net(args.batch_size).to(device)
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), lr = args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_fc.pt")


if __name__ == '__main__':
    main()
