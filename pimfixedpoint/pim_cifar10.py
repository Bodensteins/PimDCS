from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pimlinear as pl
from torchvision import datasets, transforms
import torchvision
from torch.optim.lr_scheduler import StepLR
import pimconv as pc
from quantization import de_quantization, to_float
import pim_optimizer as po
import torch.utils.data

from quantization import *


class PimVGGNet(nn.Module):
    def __init__(self, batch_size, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.conv1 = pc.PimConv2D([3, 32, 32], [3, 3], 128, batch_size, padding=1, device=device)
        self.conv2 = pc.PimConv2D([128, 32, 32], [3, 3], 128, batch_size, padding=1, device=device)
        self.conv3 = pc.PimConv2D([128, 16, 16], [3, 3], 256, batch_size, padding=1, device=device)
        self.conv4 = pc.PimConv2D([256, 16, 16], [3, 3], 256, batch_size, padding=1, device=device)
        self.conv5 = pc.PimConv2D([256, 8, 8], [3, 3], 512, batch_size, padding=1, device=device)
        self.conv6 = pc.PimConv2D([512, 8, 8], [3, 3], 512, batch_size, padding=1, device=device)
        self.conv7 = pc.PimConv2D([512, 4, 4], [3, 3], 1024, batch_size, padding=1, device=device)
        self.fc1 = pl.PimLinear(4096, 128, batch_size, device=device)
        self.relu = pl.PimRelu()
        self.fc2 = pl.PimLinear(128, 10, batch_size, device=device)
        self.dequan = pl.DeQuanLayer(16)

    def forward(self, x):
        x, x_f = pl.quanFunction.apply(x, 16)
        
        x, x_f, _ = self.conv1(x, x_f)
        x, x_f, _ = pl.ReluFunction.apply(x, x_f)
        _, _, x = self.conv2(x, x_f)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x, x_f = pl.quanFunction.apply(x, 16)

        x, x_f, _ = self.conv3(x, x_f)
        x, x_f, _ = pl.ReluFunction.apply(x, x_f)
        _, _, x = self.conv4(x, x_f)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x, x_f = pl.quanFunction.apply(x, 16)
        
        x, x_f, _ = self.conv5(x, x_f)
        x, x_f, _ = pl.ReluFunction.apply(x, x_f)
        _, _, x = self.conv6(x, x_f)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x, x_f = pl.quanFunction.apply(x, 16)

        _, _, x = self.conv7(x, x_f)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # fc layer
        x = torch.flatten(x, 1)
        x, x_f = pl.quanFunction.apply(x, 16)
        x, x_f = self.fc1(x, x_f)
        x, x_f = self.relu(x, x_f)
        x, x_f = self.fc2(x, x_f)
        x = self.dequan(x, x_f)

        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_log_interval = 0.0
    done_data = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss_value = loss.item()
        loss_log_interval += loss_value
        loss.backward()
        optimizer.step()
        done_data += len(data)

        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {:3d} [{:5d}/{:5d} ({:.2f}%)]\t Average Loss: {:.6f}'.format(
                epoch, done_data, len(train_loader.dataset), 100. * done_data / len(train_loader.dataset),
                loss_log_interval / args.log_interval))
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
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--lr-decay-step', type=int, default=30, metavar='M',
                        help='Period of learning rate decay. (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cuda_use_num', type=int, default=1, metavar='N',
                        help='use which cuda')
 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+str(args.cuda_use_num) if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = PimVGGNet(args.train_batch_size, device=device).to(device)
    optimizer = po.PimSGD(model.named_parameters(), lr=args.lr, momentum=0.9, run_mode=po.OptimMode.full_fix)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_vgg.pt")


if __name__ == "__main__":
    main()
