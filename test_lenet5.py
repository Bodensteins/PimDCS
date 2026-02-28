import argparse
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from src.utils import common
from src.utils import statistic
from src.nn_models.lenet5_model import LeNet5


def nn_train():
    # Test settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    
    # batch
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--round', type=int, default=5, metavar='ROUND',
                        help='round end')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    
    # model
    # parser.add_argument('--net', default="alexnet", metavar='NET',
    #                     help='use which NN model')
    # parser.add_argument('--model-dir', default='/home/leitaoming/Data/FADESim_data/model_weight', metavar='MD',
    #                     help='dir of load/save model')
    
    # dataset
    parser.add_argument('--dataset', default="cifar10", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--data-dir', default='/home/leitaoming/Data/FADESim_data/dataset', metavar='DD',
                        help='dir of dataset')
    
    # statstic
    parser.add_argument('--save-dir', default='/home/leitaoming/Data/FADESim_data/exp_data/train/vgg11_cifar10', metavar='MD',
                        help='dir of statistic')
    parser.add_argument('--postfix', default='simple', metavar='MD',
                        help='postfix of statistic file')

    # type
    parser.add_argument('--mmm-type', action='store_true', default='dcs',
                        help='matMulManager type: float-point, fixed-point, dcs, tailor, trq')
    
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
    
    # print(args.model_dir)

    model_04_0 = LeNet5().to(device)
    model_04_1 = LeNet5().to(device)
    model_59_0 = LeNet5().to(device)
    model_59_1 = LeNet5().to(device)
    model_load_filename_04 = '/home/leitaoming/Data/FADESim_data/model_weight/lenet5_04.pth'
    model_load_filename_59 = '/home/leitaoming/Data/FADESim_data/model_weight/lenet5_59.pth'
    dataset_dir = '/home/leitaoming/Data/FADESim_data/dataset'

    # load weight
    model_04_0.load_state_dict(torch.load(model_load_filename_04, weights_only=True))
    model_04_1.load_state_dict(torch.load(model_load_filename_04, weights_only=True))
    model_59_0.load_state_dict(torch.load(model_load_filename_59, weights_only=True))
    model_59_1.load_state_dict(torch.load(model_load_filename_59, weights_only=True))

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # train_datasets = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    test_datasets = datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)

    # 分离数据
    # train_indices_04 = [i for i, label in enumerate(train_datasets.targets) if label < 5]
    # train_indices_59 = [i for i, label in enumerate(train_datasets.targets) if label >= 5]
    test_indices_04 = [i for i, label in enumerate(test_datasets.targets) if label < 5]
    test_indices_59 = [i for i, label in enumerate(test_datasets.targets) if label >= 5]
    
    from torch.utils.data import Subset
    # train_04 = Subset(train_datasets, train_indices_04)
    # train_59 = Subset(train_datasets, train_indices_59)
    test_04 = Subset(test_datasets, test_indices_04)
    test_59 = Subset(test_datasets, test_indices_59)
    
    # 数据加载器
    # train_loader_04 = torch.utils.data.DataLoader(train_04, batch_size=args.train_batch_size, shuffle=True)
    # train_loader_59 = torch.utils.data.DataLoader(train_59, batch_size=args.train_batch_size, shuffle=True)
    test_loader_04 = torch.utils.data.DataLoader(test_04, batch_size=args.test_batch_size, shuffle=False)
    test_loader_59 = torch.utils.data.DataLoader(test_59, batch_size=args.test_batch_size, shuffle=False)

    # train_loader, test_loader, _ = common.split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs)

    # optimizer_04 = optim.SGD(model_04.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    # optimizer_59 = optim.SGD(model_59.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # if args.scheduler:
    #     scheduler_04 = optim.lr_scheduler.StepLR(optimizer_04, step_size=args.lr_decay_step, gamma=args.gamma)
    #     scheduler_59 = optim.lr_scheduler.StepLR(optimizer_59, step_size=args.lr_decay_step, gamma=args.gamma)
    # else:
    #     scheduler_04 = None
    #     scheduler_59 = None

    # criterion = nn.CrossEntropyLoss()
    # _, _, _ = common.train_model(model_04, device, train_loader_04, test_loader_04, criterion, optimizer_04, args.epochs,
    #                 model_filename=model_save_filename_04, score_type='accuracy', patience=300, scheduler=scheduler_04)
    # _, _, _ = common.train_model(model_59, device, train_loader_59, test_loader_59, criterion, optimizer_59, args.epochs,
    #                 model_filename=model_save_filename_59, score_type='accuracy', patience=300, scheduler=scheduler_59)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    print("model_04, test_04")
    common.test_model(model_04_0, device, test_loader_04, criterion, is_wrapper=True, round_end=args.round)
    statistic.save_statistic(model_04_0, args.save_dir, "simple_04_04")

    print("model_04, test_59")
    common.test_model(model_04_1, device, test_loader_59, criterion, is_wrapper=True, round_end=args.round)
    statistic.save_statistic(model_04_1, args.save_dir, "simple_04_59")

    print("model_59, test_04")
    common.test_model(model_59_0, device, test_loader_04, criterion, is_wrapper=True, round_end=args.round)
    statistic.save_statistic(model_59_0, args.save_dir, "simple_59_04")

    print("model_59, test_59")
    common.test_model(model_59_1, device, test_loader_59, criterion, is_wrapper=True, round_end=args.round)
    statistic.save_statistic(model_59_1, args.save_dir, "simple_59_59")


if __name__ == "__main__":
    nn_train()

