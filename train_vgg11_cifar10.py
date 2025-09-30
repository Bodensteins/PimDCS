import argparse
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torchvision
from src.utils import common
import torch.optim as optim
from src.nn_models.vgg_model import vgg11_cifar10


def nn_train():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--round', type=int, default=1000, metavar='ROUND',
                        help='round end')
    
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
    
    # cuda
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--cuda-use-num', type=int, default=2, metavar='CUDA',
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

    model = vgg11_cifar10().to(device)
    model_save_filename = '/home/leitaoming/Data/FADESim_data/model_weight/vgg11_cifar10_2.pth'
    dataset_dir = '/home/leitaoming/Data/FADESim_data/dataset'

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
        torchvision.datasets.CIFAR10(dataset_dir, train=True, download=False, transform=transform_train)

    test_datasets = \
        torchvision.datasets.CIFAR10(dataset_dir, train=False, download=False, transform=transform_test)

    train_loader, test_loader, _ = common.split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    _, _, _ = common.train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                    model_filename=model_save_filename, score_type='accuracy', patience=300, scheduler=scheduler)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    common.test_model(model, device, test_loader, criterion, round_end=args.round)


if __name__ == "__main__":
    nn_train()

