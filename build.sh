#!/bin/bash
build_dir=build

if [ ! -d "$build_dir" ]; then
    mkdir $build_dir
fi

#TORCH_LIBRARY=/home/bing/SSD/softwares/libtorch_gpu-1.9/libtorch
TORCH_LIBRARY=/etc/libtorch
cd $build_dir

cmake .. -DCMAKE_PREFIX_PATH=$TORCH_LIBRARY -DCMAKE_BUILD_TYPE=Debug
make -j

