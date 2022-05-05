from __future__ import print_function
import argparse
import sys

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision

import trainCommon
from fixedPoint.optim import pim_optimizer as po
import torch.utils.data

from fixedPoint.fixedPointArithmetic import *
from trainCommon import create_datasets, train_model, draw_loss_figure, test_model, \
    train_full_data, create_full_train_loader
from networkModel import VGG, PimVGG, ConvCifar10, VGG16, VGG8B


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=30, metavar='STEP',
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
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load the  trained model')
    parser.add_argument('--load-filename', default='123.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=True,
                        help='train the model')
    parser.add_argument('--pim', action='store_true', default=False,
                        help='For use pim')
    parser.add_argument('--net', type=int, default=3, metavar='NET',
                        help='use which model (0:VGG8 1:Conv 2:VGG16 3:VGG8B)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=1, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')
 
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.cuda_use_num) if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_cuda_kwargs = {'num_workers': 2, 'pin_memory': False, 'shuffle': True}
        test_cuda_kwargs = {'num_workers': 2, 'pin_memory': False, 'shuffle': False}
        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    extra_train_data = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
    test_data = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader, test_loader, valid_loader = create_datasets(train_data, test_data, train_kwargs, test_kwargs,
                                                              seed=args.seed, extra_train_data=extra_train_data)
    full_train_loader = create_full_train_loader(train_data, train_kwargs)

    # labels_map = {
    #     0: "plane",
    #     1: "car",
    #     2: "bird",
    #     3: "cat",
    #     4: "deer",
    #     5: "dog",
    #     6: "frog",
    #     7: "horse",
    #     8: "ship",
    #     9: "truck",
    # }
    if args.pim:
        if args.net == 0:
            model = PimVGG(args.train_batch_size, device=device).to(device)
            optimizer = po.PimSGD(model.named_parameters(), lr=args.lr)
        else:
            print("undefined net!")
            sys.exit()
    else:
        if args.net == 0:
            model = VGG().to(device)
            # optimizer = optim.Adadelta(model.parameters())  # 62.40%
            # optimizer = optim.Adam(model.parameters())  # 72.62%
            # optimizer = optim.Adagrad(model.parameters())  # 52.75%
            # optimizer = optim.SGD(model.parameters(), lr=0.01)  # 29.36%
            optimizer = optim.SGD(model.parameters(), lr=args.lr)  # 61.04%
            # optimizer = optim.SGD(model.parameters(), lr=0.2)  # 66.81%
        elif args.net == 1:
            model = ConvCifar10().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        elif args.net == 2:
            model = VGG16().to(device)
            optimizer = optim.Adam(model.parameters())
            # optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.net == 3:
            model = VGG8B().to(device)
            # optimizer = optim.Adam(model.parameters())
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        else:
            print("undefined net!")
            sys.exit()

    model_name = type(model).__name__

    model_file = args.model_dir + '/' + model_name + '_checkpoint.pt'

    if args.load_model:
        try:
            para = torch.load(args.load_filename)
            model.load_state_dict(para)
        except Exception as e:
            print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch / 20))
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()
        lr = trainCommon.get_optimal_learning_rate(model, device, full_train_loader, test_loader, criterion)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_file, score_type='loss',
        #                                      scheduler=scheduler)
        train_loss, valid_loss = train_model(model, device, full_train_loader, test_loader, criterion, optimizer,
                                             args.epochs, filename=model_file, score_type='accuracy',
                                             scheduler=scheduler)
        draw_loss_figure(train_loss, valid_loss, args.figure_dir + '/' + model_name + '_')

        # print("retrain use full data")
        #
        # train_loss, valid_loss = train_model(model, device, full_train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_file)
        # draw_loss_figure(train_loss, valid_loss, args.figure_dir + '/' + model_name + '_retrain_')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
