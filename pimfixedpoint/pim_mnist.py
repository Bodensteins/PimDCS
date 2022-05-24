from __future__ import print_function

import sys

import torch.utils.data
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from fixedPoint import optim as fpOptim
from trainCommon import create_datasets, train_model, test_model, draw_loss_figure
from mnist_model import PimFcMnist, FixedPointSimpleConvNet, FcMnist, ConvMnist, PimConvMnist


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0125, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--lr-decay-step', type=int, default=5, metavar='STEP',
                        help='Period of learning rate decay. (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='GAMMA',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='model', metavar='MD',
                        help='dir of load/save model')
    parser.add_argument('--figure-dir', default='result_fig', metavar='FD',
                        help='dir of save figure')
    parser.add_argument('--model-file', default='checkpoint.pt', metavar='MF',
                        help='filename of load/save model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train', action='store_true', default=True,
                        help='train the model')
    parser.add_argument('--pim', action='store_true', default=True,
                        help='For use pim')
    parser.add_argument('--net', type=int, default=0, metavar='NET',
                        help='use which model (0:conv 1:fc)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=1, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': True}
        test_cuda_kwargs = {'num_workers': 1, 'pin_memory': False, 'shuffle': False}
        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader, test_loader, valid_loader = \
        create_datasets(train_data, test_data, train_kwargs, test_kwargs, seed=args.seed)

    if args.pim:
        if args.net == 0:
            model = PimConvMnist(args.train_batch_size, device=device).to(device)
            # model = FixedPointSimpleConvNet(args.train_batch_size, device=device).to(device)
            # model.double()
        elif args.net == 1:
            model = PimFcMnist(args.train_batch_size, device=device).to(device)
        else:
            print("undefined net!")
            sys.exit()
        optimizer = fpOptim.SGD(model.named_parameters(), lr=args.lr, momentum=0.9, run_mode=fpOptim.OptimMode.full_fix)
    else:
        if args.net == 0:
            model = ConvMnist().to(device)
        elif args.net == 1:
            model = FcMnist().to(device)
        else:
            print("undefined net!")
            sys.exit()
        # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print(model)

    model_name = type(model).__name__
    model_file = args.model_dir + '/' + model_name + '_' + args.model_file

    if args.load_model:
        try:
            para = torch.load(model_file)
            model.load_state_dict(para)
        except Exception as e:
            print(e)

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    if args.train:
        train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
                                             args.epochs, filename=model_file, scheduler=scheduler)
        draw_loss_figure(train_loss, valid_loss, args.figure_dir + '/' + model_name + '_')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion)


if __name__ == '__main__':
    main()
