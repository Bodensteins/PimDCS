import argparse
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torchvision
from src.utils import common
import torch.optim as optim
from src.utils import statistic
from src.nn_models.vgg_model import vgg11_cifar10


def train_and_test(model, args, device, train_loader, valid_loader, criterion, optimizer, scheduler=None):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    valid_acc_list = []

    for epoch in range(args.epochs):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (config update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())


        ######################
        # validate the model #
        ######################
        correct = 0
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = sum(train_losses)/len(train_losses)
        valid_loss = sum(valid_losses)/len(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        valid_acc = correct / len(valid_loader.dataset)

        valid_acc_list.append(valid_acc)

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.4f} ' +
                     f'valid_loss: {valid_loss:.4f} ' +
                     f'valid acc: {100. * valid_acc:.2f}%')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if scheduler is not None:
            scheduler.step()

        print("fixed-point test")
        import copy
        model_copy = copy.deepcopy(model)
        # model_copy = vgg11_cifar10().to(device)
        # model_copy.load_state_dict(model.state_dict())
        common.test_model(model_copy, device, valid_loader, criterion, round_end=args.round, is_wrapper=True)
        
        if epoch % args.sample_epoch == 0:
            if args.postfix == "simp" or args.postfix == "test":
                print(f"----save epoch{epoch} statistic----")
                statistic.save_statistic(model_copy, args.save_dir, args.postfix + f"_epoch{epoch}")

    return avg_train_losses, avg_valid_losses, valid_acc_list


def nn_train():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--round', type=int, default=5, metavar='ROUND',
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
    
    # statstic
    parser.add_argument('--save-dir', default='/home/leitaoming/Data/FADESim_data/exp_data/train/vgg11_cifar10', metavar='MD',
                        help='dir of statistic')
    parser.add_argument('--postfix', default='simp', metavar='MD',
                        help='postfix of statistic file')
    parser.add_argument('--sample-epoch', default=20, metavar='MD',
                        help='sample epoch')

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
    # model_save_filename = '/home/leitaoming/Data/FADESim_data/model_weight/vgg11_cifar10_2.pth'
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
    
    _, _, _ = train_and_test(model, args, device, train_loader, test_loader, criterion, optimizer, scheduler=scheduler)


if __name__ == "__main__":
    nn_train()

