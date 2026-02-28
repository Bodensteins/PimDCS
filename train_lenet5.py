import argparse
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from src.utils import common
import torch.optim as optim
from src.nn_models.lenet5_model import LeNet5


def nn_train():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--round', type=int, default=10, metavar='ROUND',
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

    model_04 = LeNet5().to(device)
    model_59 = LeNet5().to(device)
    model_save_filename_04 = '/home/leitaoming/Data/FADESim_data/model_weight/lenet5_04.pth'
    model_save_filename_59 = '/home/leitaoming/Data/FADESim_data/model_weight/lenet5_59.pth'
    dataset_dir = '/home/leitaoming/Data/FADESim_data/dataset'

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_datasets = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    test_datasets = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)

    # 分离数据
    train_indices_04 = [i for i, label in enumerate(train_datasets.targets) if label < 5]
    train_indices_59 = [i for i, label in enumerate(train_datasets.targets) if label >= 5]
    test_indices_04 = [i for i, label in enumerate(test_datasets.targets) if label < 5]
    test_indices_59 = [i for i, label in enumerate(test_datasets.targets) if label >= 5]
    
    from torch.utils.data import Subset
    train_04 = Subset(train_datasets, train_indices_04)
    train_59 = Subset(train_datasets, train_indices_59)
    test_04 = Subset(test_datasets, test_indices_04)
    test_59 = Subset(test_datasets, test_indices_59)
    
    # 数据加载器
    train_loader_04 = torch.utils.data.DataLoader(train_04, batch_size=args.train_batch_size, shuffle=True)
    train_loader_59 = torch.utils.data.DataLoader(train_59, batch_size=args.train_batch_size, shuffle=True)
    test_loader_04 = torch.utils.data.DataLoader(test_04, batch_size=args.test_batch_size, shuffle=False)
    test_loader_59 = torch.utils.data.DataLoader(test_59, batch_size=args.test_batch_size, shuffle=False)

    # train_loader, test_loader, _ = common.split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

    optimizer_04 = optim.SGD(model_04.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer_59 = optim.SGD(model_59.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.scheduler:
        scheduler_04 = optim.lr_scheduler.StepLR(optimizer_04, step_size=args.lr_decay_step, gamma=args.gamma)
        scheduler_59 = optim.lr_scheduler.StepLR(optimizer_59, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler_04 = None
        scheduler_59 = None

    criterion = nn.CrossEntropyLoss()
    _, _, _ = common.train_model(model_04, device, train_loader_04, test_loader_04, criterion, optimizer_04, args.epochs,
                    model_filename=model_save_filename_04, score_type='accuracy', patience=300, scheduler=scheduler_04)
    _, _, _ = common.train_model(model_59, device, train_loader_59, test_loader_59, criterion, optimizer_59, args.epochs,
                    model_filename=model_save_filename_59, score_type='accuracy', patience=300, scheduler=scheduler_59)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    print("model_04, test_04")
    common.test_model(model_04, device, test_loader_04, criterion, is_wrapper=False, round_end=args.round)
    print("model_04, test_59")
    common.test_model(model_04, device, test_loader_59, criterion, is_wrapper=False, round_end=args.round)
    print("model_59, test_04")
    common.test_model(model_59, device, test_loader_04, criterion, is_wrapper=False, round_end=args.round)
    print("model_59, test_59")
    common.test_model(model_59, device, test_loader_59, criterion, is_wrapper=False, round_end=args.round)


if __name__ == "__main__":
    nn_train()

