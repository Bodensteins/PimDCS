from __future__ import print_function
import argparse
import sys

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision

from fixedPoint.optim import pim_optimizer as po
import torch.utils.data

from fixedPoint.nn.fixedPointArithmetic import *
from trainCommon import create_datasets, train_model, draw_loss_figure, test_model, \
    create_full_train_loader
from networkModel import vgg13, FixedPointVGG13, FixedPointVGG8B, VGG8B


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=20, metavar='STEP',
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
    parser.add_argument('--pim', action='store_true', default=True,
                        help='For use pim')
    parser.add_argument('--net', type=int, default=1, metavar='NET',
                        help='use which NN model (0:VGG 1:VGG8b)')
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
            model = FixedPointVGG13(args.train_batch_size, device=device).to(device)
            optimizer = po.PimSGD(model.named_parameters(), lr=args.lr)
        elif args.net == 1:
            model = FixedPointVGG8B(args.train_batch_size, device=device).to(device)
            optimizer = po.PimSGD(model.named_parameters(), lr=args.lr)
        else:
            print("undefined net!")
            sys.exit()
    else:
        if args.net == 0:
            model = vgg13(batch_norm=True).to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
        elif args.net == 1:
            model = VGG8B().to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        else:
            print("undefined net!")
            sys.exit()

    print(model)

    model_name = type(model).__name__

    model_file = args.model_dir + '/' + model_name + '_checkpoint.pt'

    if args.load_model:
        try:
            para = torch.load(args.load_filename)
            model.load_state_dict(para)
        except Exception as e:
            print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
        # lr become gamma of it every lr_decay_step
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()
        # lr = trainCommon.get_optimal_learning_rate(model, device, full_train_loader, test_loader, criterion)
        # lr = args.lr
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        # train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_file, score_type='loss',
        #                                      scheduler=scheduler)
        train_loss, valid_loss = train_model(model, device, full_train_loader, test_loader, criterion, optimizer,
                                             args.epochs, filename=model_file, score_type='accuracy', patience=60,
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
