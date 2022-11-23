import argparse
import time

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
import json

from fixedPoint import optim as fpOptim
import torch.utils.data

from fixedPoint.nn.fixedPointArithmetic import *
from trainCommon import split_data_loader, train_model, test_model, create_layer_weight_bit_width_list, \
    load_float_weight_for_fixed_point, draw_data_graph
from VGG_cifar10_model import vgg16, FixedPointVGG8B, VGG8B, fp_vgg16, VGG16ForMotivation, fp_vgg19, fp_vgg11, fp_vgg13


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--train-batch-size', type=int, default=128, metavar='TRAIN_BATCH',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='TEST_BATCH',
                        help='input batch size for testing (default: 1000)')
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
    parser.add_argument('--data-dir', default='data', metavar='DD',
                        help='dir of dataset')
    parser.add_argument('--result-dir', default='result', metavar='RD',
                        help='dir of train result')
    parser.add_argument('--model-dir', default='model', metavar='MD',
                        help='dir of load/save model')
    parser.add_argument('--trace-dir', default='trace', metavar='TD',
                        help='dir of load/save trace')
    parser.add_argument('--load-model-type', type=int, default=2, metavar='LD',
                        help='load mode type (0:no 1:float point model 2:fixed point model')
    parser.add_argument('--load-filename', default='FixedPointVGG_VGG19_half_checkpoint.pt', metavar='LF',
                        help='filename of load model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--save-trace', action='store_true', default=False,
                        help='save all model in train process')
    parser.add_argument('--fixed-point', action='store_true', default=True,
                        help='For use fixed point')
    parser.add_argument('--mix-precision', action='store_true', default=True,
                        help='mix-precision or not')
    parser.add_argument('--half-float', action='store_true', default=True,
                        help='For use 16b float')
    parser.add_argument('--net', default="VGG19", metavar='NET',
                        help='use which NN model')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--cuda-use-num', type=int, default=2, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')
 
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

    if args.mix_precision:
        # conv_input_bit_width
        # conv_output_bit_width
        # conv_weight_bit_width
        # conv_grad_output_bit_width
        # conv_next_grad_output_bit_width
        # conv_compute_weight_bit_width
        # fc_output_bit_width
        # fc_weight_bit_width
        # fc_grad_output_bit_width
        # fc_compute_weight_bit_width
        bit_width_tuple = (16, 16, 16, 16, 16, 8, 16, 16, 16, 8)
        bit_width_type = "_half"
    else:
        bit_width_tuple = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
        bit_width_type = "_full16"

    if args.fixed_point:
        if args.net == "VGG16":
            model = fp_vgg16(*bit_width_tuple)
            model.to(device)
        elif args.net == "VGG19":
            model = fp_vgg19(*bit_width_tuple)
            model.to(device)
        elif args.net == "VGG11":
            model = fp_vgg11(*bit_width_tuple)
            model.to(device)
        elif args.net == "VGG13":
            model = fp_vgg13(*bit_width_tuple)
            model.to(device)
        else:
            raise Exception('undefined net: ' + str(args.net))

        weight_bit_width_list = create_layer_weight_bit_width_list(model)

        optimizer = fpOptim.SGD(model.named_parameters(), weight_bit_width_list, lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum)
    else:
        if args.net == "VGG16":
            model = vgg16().to(device)
            if args.half_float:
                model.half()
        elif args.net == "VGG13":
            model = vgg16().to(device)
            if args.half_float:
                model.half()
        elif args.net == "VGG11":
            model = vgg16().to(device)
            if args.half_float:
                model.half()
        elif args.net == "VGG19":
            model = vgg16().to(device)
            if args.half_float:
                model.half()
        elif args.net == "VGG8B":
            model = VGG8B().to(device)
            if args.half_float:
                model.half()
        elif args.net == "VGG16ForMotivation":
            model = VGG16ForMotivation().to(device)
            if args.half_float:
                model.half()
        else:
            raise Exception('undefined net: ' + str(args.net))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print(model)

    model_name = type(model).__name__

    # time_str = time.strftime("_%Y%m%d_%H%M%S_")

    model_save_filename = args.model_dir + '/' + model_name + '_' + args.net + bit_width_type + '_checkpoint.pt'

    model_load_filename = args.model_dir + '/' + args.load_filename

    try:
        if (args.load_model_type == 1 and not args.fixed_point) or (args.load_model_type == 2 and args.fixed_point):
            para = torch.load(model_load_filename)
            model.load_state_dict(para)
        elif args.load_model_type == 1 and args.fixed_point:
            load_float_weight_for_fixed_point(model_load_filename, model)
        elif args.load_model_type:
            print("unsupported load type, load_model_type: " + str(args.load_model_type)
                  + ", but fixed point: " + str(args.fixed_point))
    except Exception as e:
        print(e)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    else:
        scheduler = None

    if args.train:
        criterion = nn.CrossEntropyLoss()

        if args.save_trace:
            trace_filename = args.trace_dir + '/' + model_name + '_' + args.net + bit_width_type + '/'
        else:
            trace_filename = "trace/other_network/"
        # train_loss, valid_loss = train_model(model, device, train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename, score_type='loss',
        #                                      scheduler=scheduler)
        train_loss_list, valid_loss_list, valid_acc_list\
            = train_model(model, device, train_loader, test_loader, criterion, optimizer, args.epochs,
                          model_filename=model_save_filename, score_type='accuracy', patience=300, scheduler=scheduler,
                          half=args.half_float, save_trace=args.save_trace, trace_filename=trace_filename)

        result_dir = args.result_dir + '/' + model_name + '_' + args.net + bit_width_type

        with open(result_dir + '_train_loss.out', 'w') as FD:
            FD.write(json.dumps(train_loss_list))

        with open(result_dir + '_valid_loss.out', 'w') as FD:
            FD.write(json.dumps(valid_loss_list))

        with open(result_dir + '_valid_acc.out', 'w') as FD:
            FD.write(json.dumps(valid_acc_list))

        # print("retrain use full data")
        #
        # train_loss, valid_loss = train_model(model, device, full_train_loader, valid_loader, criterion, optimizer,
        #                                      args.epochs, filename=model_save_filename)

    # y_lists = [(model.conv2InputList, dict(color="red", label="conv2.activation")),
    #            (model.conv3InputList, dict(color="green", label="conv3.activation")),
    #            (model.fc2InputList, dict(color="blue", label="fc2.activation")),
    #            (model.fc3InputList, dict(color="orange", label="fc3.activation"))]
    # z_lists = [(model.conv2WeightList, dict(color="red", label="conv2.weight")),
    #            (model.conv3WeightList, dict(color="green", label="conv3.weight")),
    #            (model.fc2WeightList, dict(color="blue", label="fc2.weight")),
    #            (model.fc3WeightList, dict(color="orange", label="fc3.weight"))]
    # x_list = list(range(len(model.conv2InputList)))
    # draw_data_graph(x_list, y_lists, x_label="iteration * 1000", y_label="MaxValue(log2)", title="Activation")
    # draw_data_graph(x_list, z_lists, x_label="iteration * 1000", y_label="MaxValue(log2)", title="Weight")

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_model(model, device, test_loader, criterion, half=args.half_float)


if __name__ == "__main__":
    main()
