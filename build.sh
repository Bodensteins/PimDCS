#!/bin/bash
build_dir=build

if [ ! -d "$build_dir" ]; then
    mkdir $build_dir
fi

TORCH_LIBRARY=/opt/libtorch
cd ./build

cmake .. -DCMAKE_PREFIX_PATH=$TORCH_LIBRARY -DCMAKE_BUILD_TYPE=Release
make -j

