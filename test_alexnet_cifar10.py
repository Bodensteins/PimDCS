import argparse
import time

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
# import json

import src.pimtorch.optim as pimOptim
import torch.utils.data

from src.pimtorch.nn.fixedPointArithmetic import *
from examples.trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point
# from examples.vgg_cifar10_model import vgg16, vgg11, vgg13, vgg19, fp_vgg16, VGG16ForMotivation, fp_vgg19, fp_vgg11, fp_vgg13

from examples.alexnet_ou_model import PimAlexNet_OU

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='use scheduler or not')
    parser.add_argument('--lr-decay-step', type=int, default=30, metavar='STEP',
                        help='Period of learning rate decay. (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='GAMMA',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--result-dir', default='result', metavar='RD',
                        help='dir of train result') 
    parser.add_argument('--trace-dir', default='trace', metavar='TD',
                        help='dir of load/save trace')
    parser.add_argument('--save-trace', action='store_true', default=False,
                        help='save all model in train process')
    parser.add_argument('--save-result', action='store_true', default=False,
                        help='save loss and acc')
    parser.add_argument('--mix-precision', action='store_true', default=False,
                        help='mix-precision or not')
    parser.add_argument('--half-float', action='store_true', default=False,
                        help='For use 16b float')
    parser.add_argument('--net', default="VGG11_OU", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    
    parser.add_argument('--data-dir', default='/home/leitaoming/data/pimtorch_data/dataset', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='/home/leitaoming/data/pimtorch_data/model_weight', metavar='MD',
                        help='dir of load/save model')
    # parser.add_argument('--data-dir', default='data', metavar='DD',
    #                     help='dir of dataset')
    # parser.add_argument('--model-dir', default='model', metavar='MD',
    #                     help='dir of load/save model')
    parser.add_argument('--load-model-type', type=int, default=2, metavar='LD',
                        help='load mode type (0:no 1:float point model 2:fixed point model')
    parser.add_argument('--weight-filename', default='PimAlexNet_OU_cifar10_checkpoint.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--fixed-point', action='store_true', default=True,
                        help='For use fixed point')
    parser.add_argument('--cuda-use-num', type=int, default=0, metavar='CUDA',
                        help='use which cuda (choice: 0-3)')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

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

    train_datasets = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)

    test_datasets = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)

    train_loader, test_loader, _ = split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

    model = PimAlexNet_OU().to(device)

    if args.fixed_point:
        weight_bit_width_list = create_layer_weight_bit_width_list(model)
        optimizer = pimOptim.SGD(model.named_parameters(), weight_bit_width_list, lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print(model)

    model_save_filename = args.model_dir + '/' + args.weight_filename
    model_load_filename = args.model_dir + '/' + args.weight_filename

    print(model_save_filename)
    print(model_load_filename)

    try:
        if (args.load_model_type == 1 and not args.fixed_point) or (args.load_model_type == 2 and args.fixed_point):
            para = torch.load(model_load_filename, weights_only=True)
            model.load_state_dict(para)
        elif args.load_model_type == 1 and args.fixed_point:
            load_float_weight_for_fixed_point(model_load_filename, model)
        elif args.load_model_type:
            print("unsupported load type, load_model_type: " + str(args.load_model_type)
                + ", but fixed point: " + str(args.fixed_point))
    except Exception as e:
        print(e)

    if args.train:
        if args.scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()
        _,_,_= train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                        model_filename=model_save_filename, score_type='accuracy', patience=300, scheduler=scheduler,
                        half=args.half_float)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion, half=args.half_float)

    if args.fixed_point and not args.train:
        model.print_statistic()

if __name__ == "__main__":
    main()
