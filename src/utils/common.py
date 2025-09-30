import torch
from src.utils import wrapper
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import Generator

def load_model_and_dataset(model_name, dataset_name, weight_path, dataset_path, device):
    # load model
    if model_name == 'alexnet':
        from src.nn_models.alexnet_model import AlexNet
        model = AlexNet().to(device)
    elif model_name == 'vgg11':
        from src.nn_models.vgg_model import vgg11_cifar10
        model = vgg11_cifar10().to(device)
    elif model_name == 'vgg16':
        from src.nn_models.vgg_model import vgg16_imagenet
        model = vgg16_imagenet().to(device)
    elif model_name == 'resnet18':
        from src.nn_models.resnet_model import resnet18
        model = resnet18().to(device)
    elif model_name == 'googlenet':
        from src.nn_models.googlenet_model import GoogleNet
        model = GoogleNet().to(device)
    else:
        raise Exception(f"model {model_name} not support")
    model_load_filename = weight_path +'/' + model_name + '_' + dataset_name + '.pth'
    print(model_load_filename)
    try:
        para = torch.load(model_load_filename, weights_only=True)
        model.load_state_dict(para)
    except Exception as e:
        print(e)

    # load dataset
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform)
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    else:
        raise Exception(f"dataset {dataset_name} not support")
    
    return model, dataset

def test_model(model, device, test_loader, criterion, round_end=10):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    correct = 0

    wrapper.wrap_modules(model)
    print(model)

    model.eval()  # prep model for evaluation

    round = 0
    data_len = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_len = len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            # calculate the loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            round += 1
            print("round: {}, loss: {:.4f}, acc: {:.4f}".format(round, test_loss, correct / round / data_len))
            
            if round == round_end:
                break

    # calculate and print avg test loss
    # test_loss /= len(test_loader.dataset)

    tot = round * data_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, tot, 100. * correct / tot))
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, filename, patience=7, verbose=False, delta=0, score_type='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.filename = filename
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.val_acc_max = 0.0
        self.delta = delta
        self.score_type = score_type

    def __call__(self, score_indicator, model):
        if self.score_type == 'accuracy':
            score = score_indicator
        else:  # loss
            score = -score_indicator

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score_indicator, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score_indicator, model)
            self.counter = 0

    def save_checkpoint(self, score_indicator, model):
        """Saves model when score become better."""
        torch.save(model.state_dict(), self.filename)

        if self.score_type == 'accuracy':
            if self.verbose:
                print(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {score_indicator:.4f}). '
                      f'Saving model ...')
            self.val_acc_max = score_indicator
            pass
        else:  # loss
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {score_indicator:.4f}). '
                      f'Saving model ...')
            self.val_loss_min = score_indicator


def split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs, alpha=0.2, seed=1,
                      extra_train_datasets_for_valid=None):
    if extra_train_datasets_for_valid is None:
        train_loader = DataLoader(train_datasets, **train_kwargs)
        valid_loader = None
    else:
        full_train_size = len(train_datasets)
        valid_size = int(full_train_size * alpha)
        train_size = full_train_size - valid_size

        sub_train_datasets, _ = random_split(
            dataset=train_datasets,
            lengths=[train_size, valid_size],
            generator=Generator().manual_seed(seed)
        )

        _, sub_valid_datasets = random_split(
            dataset=extra_train_datasets_for_valid,
            lengths=[train_size, valid_size],
            generator=Generator().manual_seed(seed)
        )

        train_loader = DataLoader(sub_train_datasets, **train_kwargs, shuffle=True)
        valid_loader = DataLoader(sub_valid_datasets, **test_kwargs)

    test_loader = DataLoader(test_datasets, **test_kwargs)

    return train_loader, test_loader, valid_loader


def train_model(model, device, train_loader, valid_loader, criterion, optimizer, n_epochs, patience=20,
                model_filename='model/network_checkpoint.pth', verbose=True, score_type='loss', scheduler=None,
                save_trace=False, trace_filename="trace/other_network/"):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    valid_acc_list = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(filename=model_filename, patience=patience, verbose=verbose, score_type=score_type)

    mid_batch_of_epoch = train_loader.__len__() // 2

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader):
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

            if save_trace:
                if batch == mid_batch_of_epoch:
                    torch.save(model.state_dict(), trace_filename + "epoch_" + str(epoch) + "_before_mid.pt")
                elif batch == mid_batch_of_epoch + 1:
                    torch.save(model.state_dict(), trace_filename + "epoch_" + str(epoch) + "_after_mid.pt")

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

        epoch_len = len(str(n_epochs))

        valid_acc = correct / len(valid_loader.dataset)

        valid_acc_list.append(valid_acc)

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.4f} ' +
                     f'valid_loss: {valid_loss:.4f} ' +
                     f'valid acc: {100. * valid_acc:.2f}%')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if scheduler is not None:
            scheduler.step()

        # if save_trace is True:
        #     torch.save(model.state_dict(), "epoch" + str(epoch) + filename)

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        if score_type == 'accuracy':
            early_stopping(valid_acc, model)
        else:
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(model_filename))

    return avg_train_losses, avg_valid_losses, valid_acc_list