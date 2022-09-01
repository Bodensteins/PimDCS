import argparse

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision

from fixedPoint import optim as fpOptim
import torch.utils.data

from fixedPoint.nn.fixedPointArithmetic import *
from trainCommon import split_data_loader, train_model, test_model, create_layer_bit_width_list, show_data_img
from VGG_cifar10_model import vgg16, vgg19, FixedPointVGG8B, VGG8B, fp_vgg19, fp_vgg11, fp_vgg16


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='model', metavar='MD',
                        help='dir of load/save model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load the trained model')
    parser.add_argument('--load-filename', default='123.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=True,
                        help='train the model')
    parser.add_argument('--fixed-point', action='store_true', default=True,
                        help='For use fixed point')
    parser.add_argument('--half-float', action='store_true', default=False,
                        help='For use 16b float')
    parser.add_argument('--net', type=int, default=0, metavar='NET',
                        help='use which NN model (0:VGG 1:VGG8b)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda_use_num', type=int, default=0, metavar='CUDA',
                        help='use which cuda (choice: 0-1)')
 
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
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)

    # extra_train_datasets_for_valid = \
    #     torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
    test_datasets = \
        torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader, test_loader, _ = split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

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

    # show_data_img(labels_map, train_datasets)
    if args.fixed_point:
        if args.net == 0:
            # model = fp_vgg19(args.train_batch_size, device=device, batch_norm=True).to(device)
            model = fp_vgg16(batch_norm=False).to(device)
            # model.double()
            # model = FixedPointVGG13(args.train_batch_size, device=device).to(device)
        elif args.net == 1:
            model = FixedPointVGG8B(args.train_batch_size, device=device).to(device)
        else:
            raise Exception('undefined net: ' + str(args.net))

        bit_width_list = create_layer_bit_width_list(model)

        optimizer = fpOptim.SGD(model.named_parameters(), bit_width_list, lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum)
    else:
        if args.net == 0:
            model = vgg16().to(device)
            model.half()
        elif args.net == 1:
            model = VGG8B().to(device)
        else:
            raise Exception('undefined net: ' + str(args.net))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print(model)

    model_name = type(model).__name__
    model_save_filename = args.model_dir + '/' + model_name + '_checkpoint.pt'

    if args.load_model:
        model_load_filename = args.model_dir + '/' + args.load_filename
        try:
            para = torch.load(model_load_filename)
            model.load_state_dict(para)
        except Exception as e:
            print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()
        # train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename, score_type='loss',
        #                                      scheduler=scheduler)
        _, _, _ = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                              filename=model_save_filename, score_type='accuracy', patience=100, scheduler=scheduler,
                              half=args.half_float)

        # print("retrain use full data")
        #
        # train_loss, valid_loss = train_model(model, device, full_train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion, half=args.half_float)


if __name__ == "__main__":
    main()
