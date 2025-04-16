import argparse
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data
from torch.utils.data import DataLoader
from src.pimtorch.nn.fixedPointArithmetic import *
from examples.trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point
from examples.googlenet_ou_model import PimGoogLeNet_OU


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
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

    parser.add_argument('--data-dir', default='/home/leitaoming/Data/FADESim_data/dataset/ImageNet', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--model-dir', default='/home/leitaoming/Data/FADESim_data/model_weight', metavar='MD',
                        help='dir of load/save model')
    # parser.add_argument('--data-dir', default='data/ImageNet', metavar='DD',
    #                     help='dir of dataset')
    # parser.add_argument('--model-dir', default='model', metavar='MD',
    #                     help='dir of load/save model')
    parser.add_argument('--load-model-type', type=int, default=2, metavar='LD',
                        help='load mode type (0:no 1:float point model 2:fixed point model')
    parser.add_argument('--weight-filename', default='PimGoogLeNet_OU_ImageNet_checkpoint.pth', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--fixed-point', action='store_true', default=True,
                        help='For use fixed point')
    parser.add_argument('--cuda-use-num', type=int, default=3, metavar='CUDA',
                        help='use which cuda (choice: 0-3)')
    args = parser.parse_args()
    # print(args)

    torch.manual_seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    test_kwargs = {'batch_size': args.test_batch_size}

    # 数据预处理，包括大小调整和标准化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=args.data_dir, transform=transform)
    test_loader = DataLoader(dataset, **test_kwargs)

    model = PimGoogLeNet_OU().to(device)
    print(model)

    # model_save_filename = args.model_dir + '/' + args.weight_filename
    model_load_filename = args.model_dir + '/' + args.weight_filename

    # print(model_save_filename)
    # print(model_load_filename)

    try:
        para = torch.load(model_load_filename, map_location=device)
        model.load_state_dict(para,strict=False)
    except Exception as e:
        print(e)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion, half=args.half_float)

    if args.fixed_point and not args.train:
        model.print_statistic()


if __name__ == "__main__":
    main()