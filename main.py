import argparse
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from src.utils import common
from src.utils import statistic


def nn_run():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    
    # batch
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--round', type=int, default=5, metavar='ROUND',
                        help='round end')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    
    # model
    parser.add_argument('--net', default="resnet18", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--model-dir', default='~/Data/FADESim_data/model_weight', metavar='MD',
                        help='dir of load/save model')
    
    # dataset
    parser.add_argument('--dataset', default="imagenet", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--data-dir', default='/mnt/hdd1/dataset/imagenet/val', metavar='DD',
                        help='dir of dataset')
    
    # statstic
    parser.add_argument('--save-dir', default='~/Data/FADESim_data/exp_data/resnet18_imagenet', metavar='MD',
                        help='dir of statistic')
    parser.add_argument('--postfix', default='test', metavar='MD',
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

    test_kwargs = {'batch_size': args.test_batch_size}
    model, dataset = common.load_model_and_dataset(args.net, args.dataset, args.model_dir, args.data_dir, device)
    test_loader = DataLoader(dataset, **test_kwargs, shuffle=True)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    common.test_model(model, device, test_loader, criterion, round_end=args.round)

    if args.postfix == 'sample' or args.postfix == 'test':
        statistic.save_statistic(model, args.save_dir, args.postfix)


if __name__ == "__main__":
    nn_run()

