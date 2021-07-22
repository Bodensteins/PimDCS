#!/bin/bash
data_dir=data

if [ ! -d "$data_dir" ]; then
    mkdir $data_dir
fi

cd $data_dir
#mnist dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
#cifar10 dataset
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
