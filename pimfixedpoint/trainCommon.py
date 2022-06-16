import torch
from matplotlib import pyplot as plt
from torch import Generator, optim
from torch.utils.data import DataLoader, random_split
from fixedPoint.nn.earlystopping import EarlyStopping


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

        train_loader = DataLoader(sub_train_datasets, **train_kwargs)
        valid_loader = DataLoader(sub_valid_datasets, **test_kwargs)

    test_loader = DataLoader(test_datasets, **test_kwargs)

    return train_loader, test_loader, valid_loader


def train_model(model, device, train_loader, valid_loader, criterion, optimizer, n_epochs,
                patience=20, filename='checkpoint.pt', verbose=True, score_type='loss', scheduler=None):
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
    early_stopping = EarlyStopping(filename=filename, patience=patience, verbose=verbose, score_type=score_type)

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
            # perform a single optimization step (parameter update)
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

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if score_type == 'accuracy':
            early_stopping(valid_acc, model)
        else:
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(filename))

    return avg_train_losses, avg_valid_losses, valid_acc_list


def train_full_data(model, device, train_loader, criterion, optimizer, n_epochs, filename='full_data.pt'):
    # to track the training loss as the model trains
    train_losses = []

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
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = sum(train_losses)/len(train_losses)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.4f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []

    # save the model
    torch.save(model.state_dict(), filename)


def test_model(model, device, test_loader, criterion=None):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    correct = 0

    model.eval()  # prep model for evaluation

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate and print avg test loss
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def draw_loss_figure(train_loss, valid_loss, model_name):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    min_pos = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(min_pos, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    figure_name = model_name + 'loss_plot.png'
    fig.savefig(figure_name, bbox_inches='tight')


def show_data_img(labels_map, data, figure_size=(32, 32), cols=3, rows=3, random=False, is_gray=False):
    figure = plt.figure(figsize=figure_size)
    for i in range(1, cols * rows + 1):
        if random:
            sample_idx = torch.randint(len(data), size=(1,)).item()
        else:
            sample_idx = i - 1
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")

        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)

        if is_gray:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)

    plt.show()


# only for sgd
def get_optimal_learning_rate(model, device, train_loader, valid_loader, criterion, filename='model/temp.pt',
                              initial_lr=0.0009765625, epoch=5, n=10, score_type='loss'):
    lr = initial_lr
    global_valid_loss_min = float('inf')
    lr_optimal = None
    i = 1

    while i <= n:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print(f"now lr is: {lr}, we try train {epoch} epochs")
        _, valid_loss, _ = train_model(model, device, train_loader, valid_loader, criterion, optimizer, epoch,
                                       patience=epoch+1, filename=filename + '_' + str(lr), verbose=False)
        local_valid_loss_min = min(valid_loss)
        if local_valid_loss_min < global_valid_loss_min:
            lr_optimal = lr
            global_valid_loss_min = local_valid_loss_min
        print(f"lr: {lr}, valid loss {local_valid_loss_min}")
        lr *= 2
        i += 1
        net_reset_parameters(model)

    return lr_optimal


def net_reset_parameters(model):
    # for layer in model.modules():
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def create_layer_bit_width_list(model):
    # for layer in model.modules():
    bit_width_list = []
    for layer in model.children():
        if hasattr(layer, 'weightBits'):
            bit_width_list.append(layer.weightBits)

    return bit_width_list
