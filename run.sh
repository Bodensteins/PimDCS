# nohup python main.py --net vgg11 --dataset cifar10 \
#     --model-dir /home/leitaoming/Data/FADESim_data/model_weight \
#     --data-dir /home/leitaoming/Data/FADESim_data/dataset \
#     --save-dir /home/leitaoming/Data/FADESim_data/exp_data/vgg11_cifar10 --postfix test \
#     --cuda-use-num 1 --round 50 --test-batch-size 20  > log/test_vgg11_cifar10.log 2>&1 &

# nohup python main.py --net vgg16 --dataset imagenet \
#     --model-dir /home/leitaoming/Data/FADESim_data/model_weight \
#     --data-dir /mnt/hdd1/dataset/imagenet/val \
#     --save-dir /home/leitaoming/Data/FADESim_data/exp_data/vgg16_imagenet --postfix sample \
#     --cuda-use-num 3 --round 10 --test-batch-size 5  > log/test_vgg16_imagenet.log 2>&1 &

nohup python main.py --net alexnet --dataset cifar10 \
    --model-dir /home/leitaoming/Data/FADESim_data/model_weight \
    --data-dir /home/leitaoming/Data/FADESim_data/dataset \
    --save-dir /home/leitaoming/Data/FADESim_data/exp_data/alexnet_cifar10 --postfix sample \
    --cuda-use-num 1 --round 10 --test-batch-size 10  > log/test_alexnet_cifar10.log 2>&1 &

# nohup python main.py --net resnet18 --dataset imagenet \
#     --model-dir /home/leitaoming/Data/FADESim_data/model_weight \
#     --data-dir /mnt/hdd1/dataset/imagenet/val \
#     --save-dir /home/leitaoming/Data/FADESim_data/exp_data/resnet18_imagenet --postfix test \
#     --cuda-use-num 2 --round 100 --test-batch-size 10  > log/test_resnet18_imagenet.log 2>&1 &
