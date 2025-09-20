#nohup python test_alexnet_cifar10.py --cuda-use-num 0 --round 5 --test-batch-size 10 --postfix sample > log/test_alexnet_cifar10.log 2>&1 &
#nohup python test_vgg11_cifar10.py  --cuda-use-num 1 --round 5 --test-batch-size 10 --postfix sample > log/test_vgg11_cifar10.log 2>&1 &
#nohup python test_vgg16_imagenet.py --cuda-use-num 3 --round 500 --test-batch-size 10 --postfix test > log/test_vgg16_imagenet_2.log 2>&1 &
#nohup python test_resnet18_imagenet.py --cuda-use-num 2 --round 100 --test-batch-size 10 --postfix sample > log/test_resnet18_imagenet.log 2>&1 &

nohup python main.py --net alexnet --dataset cifar10 \
    --model-dir /home/leitaoming/Data/FADESim_data/model_weight \
    --data-dir /home/leitaoming/Data/FADESim_data/dataset \
    --save-dir /home/leitaoming/Data/FADESim_data/exp_data/alexnet_cifar10 --postfix sample \
    --cuda-use-num 1 --round 5 --test-batch-size 10  > log/test_alexnet_cifar10.log 2>&1 &

