//
// Created by 周恒 on 2021/2/6.
//

#ifndef PIMTORCH_VGG_H
#define PIMTORCH_VGG_H
#pragma once
#include <torch/torch.h>
#include <array>
#include <vector>


torch::nn::Sequential vgg_block(const int kNumConvs,  int in_channels,  int out_channels) {
  torch::nn::Sequential vgg_layers;
  const int kKernelSize = 3;

  for (int i = 0; i < kNumConvs; ++i) {
    vgg_layers->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(/*in_channels=*/in_channels, /*out_channels=*/out_channels, kKernelSize).padding(/*padding=*/1)));
    vgg_layers->push_back(torch::nn::ReLU());
    in_channels = out_channels;

  }
  vgg_layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/{2,2}).stride(2)));
  return vgg_layers;
}

class Flatten : public torch::nn::Module {
public:
  torch::Tensor forward(const torch::Tensor& input) {
    return input.view({input.sizes()[0], -1});
  }
};

class VGG : public torch::nn::Module {
public:
  VGG(const std::vector<std::array<int, 2>>& conv_arch)
      : conv_arch_(conv_arch),
        classifier_(
            /*The fully connected layer part*/
            Flatten(),
            torch::nn::Linear(/*in_features=*/2304, /*out_features=*/4096),
            torch::nn::ReLU(),
            torch::nn::Dropout(/*p=*/0.4),
            torch::nn::Linear(/*in_features=*/4096, /*out_features=*/4096),
            torch::nn::ReLU(),
            torch::nn::Dropout(/*p=*/0.4),
            torch::nn::Linear(/*in_features=*/4096, /*out_features=*/10)
        )
  {
    const int kModuleSize = conv_arch.size();
    int in_channels = 1;
    for (int i = 0; i < kModuleSize; ++i) {
      int out_channels = conv_arch_[i][1];
      vgg_seq_layers_->extend(*vgg_block(conv_arch_[i][0], in_channels, out_channels));
      in_channels = out_channels;
    }

    VGGNetwork_->extend(*vgg_seq_layers_);
    VGGNetwork_->extend(*classifier_);
    register_module("VGGNetwork_", VGGNetwork_);

  }

  torch::Tensor forward(torch::Tensor& input) {
    return VGGNetwork_->forward(input);
  }
private:
  std::vector<std::array<int, 2>> conv_arch_;
  torch::nn::Sequential vgg_seq_layers_;
  torch::nn::Sequential classifier_;
  torch::nn::Sequential VGGNetwork_;
};

#endif //PIMTORCH_VGG_H
