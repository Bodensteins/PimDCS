import torch
from src.utils import wrapper
import torchvision
from torchvision import transforms

def load_model_and_dataset(model_name, dataset_name, weight_path, dataset_path, device):
    if model_name == 'alexnet':
        from src.nn_models.alexnet_model import AlexNet
        model = AlexNet().to(device)
    elif model_name == 'vgg11':
        from src.nn_models.vgg11_model import VGG11
        model = VGG11().to(device)
    elif model_name == 'vgg16':
        from src.nn_models.vgg_model import vgg16
        model = vgg16().to(device)
    elif model_name == 'resnet18':
        from src.nn_models.resnet_model import resnet18
        model = resnet18().to(device)
    elif model_name == 'googlenet':
        from src.nn_models.googlenet_model import GoogleNet
        model = GoogleNet().to(device)
    else:
        raise Exception(f"model {model_name} not support")
    
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
    
    model_load_filename = weight_path +'/' + model_name + '_' + dataset_name + '.pth'
    print(model_load_filename)
    try:
        para = torch.load(model_load_filename, weights_only=True)
        model.load_state_dict(para)
    except Exception as e:
        print(e)

    return model, dataset

def test_model(model, device, test_loader, criterion, half=False, round_end=10):
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
    