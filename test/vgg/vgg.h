//
// Created by 周恒 on 2021/2/6.
//

#ifndef PIMTORCH_VGG_H
#define PIMTORCH_VGG_H
#pragma once
#include <torch/torch.h>
#include <array>
#include <vector>
#include "pim_conv.h"
#include "pim_linear.h"

torch::nn::Sequential vgg_block(const int kNumConvs, int in_channels, int out_channels, bool use_pim,
    std::vector<ExpandingArray<4>>& in_shapes ) {
  torch::nn::Sequential vgg_layers;
  const int kKernelSize = 3;

  for (int i = 0; i < kNumConvs; ++i) {
    if (use_pim) {
      vgg_layers->push_back(PIM::PimConv2d(
          ExpandingArray<4>(in_shapes[i]),
          PIM::PimArrayType::simple_logic_array,
          Conv2dOptions(in_channels, out_channels, kKernelSize).padding(1)
      ));
    } else {
      vgg_layers->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(/*in_channels=*/in_channels, /*out_channels=*/out_channels, kKernelSize).padding(/*padding=*/1)
      ));
    }
    vgg_layers->push_back(torch::nn::ReLU());
    in_channels = out_channels;

  }
  vgg_layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/{2,2}).stride(2)));
  return vgg_layers;
}

class VGG : public torch::nn::Module {
public:
  VGG(const std::vector<std::array<int, 2>>& conv_arch, bool use_pim, int64_t batch_size,
      std::vector<std::vector<ExpandingArray<4>>>& in_shapes);

  torch::Tensor forward(torch::Tensor& input) {
    return VGGNetwork_->forward(input);
  }
private:
  std::vector<std::array<int, 2>> conv_arch_;
  torch::nn::Sequential vgg_seq_layers_;
  torch::nn::Sequential classifier_;
  torch::nn::Sequential VGGNetwork_;
  bool use_pim;
  int64_t batch_size;
};

VGG::VGG(const std::vector<std::array<int, 2>> &conv_arch, bool use_pim, int64_t batch_size,
    std::vector<std::vector<ExpandingArray<4>>>& in_shapes)
    : conv_arch_(conv_arch),
      use_pim(use_pim),
      batch_size(batch_size)
{
  const int kModuleSize = conv_arch.size();
  int in_channels = 3;

  if (use_pim) {
    classifier_ = torch::nn::Sequential(
        /*The fully connected layer part*/
        torch::nn::Flatten(),
        // original: 512, 3 block: 2304
        PIM::PimLinear(512, 4096, batch_size, PIM::PimArrayType::simple_logic_array),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.4),
        PIM::PimLinear(4096, 4096, batch_size, PIM::PimArrayType::simple_logic_array),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.4),
        PIM::PimLinear(4096, 10, batch_size, PIM::PimArrayType::simple_logic_array)
    );
    for (int i = 0; i < kModuleSize; ++i) {
      int out_channels = conv_arch_[i][1];
      vgg_seq_layers_->extend(
          *vgg_block(conv_arch_[i][0], in_channels, out_channels, true, in_shapes[i])
      );
      in_channels = out_channels;
    }
  } else {
    classifier_ = torch::nn::Sequential(
        /*The fully connected layer part*/
        torch::nn::Flatten(),
        // original: 512, 3 block: 2304
        torch::nn::Linear(/*in_features=*/512, /*out_features=*/4096),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.4),
        torch::nn::Linear(/*in_features=*/4096, /*out_features=*/4096),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.4),
        torch::nn::Linear(/*in_features=*/4096, /*out_features=*/10)
    );
    for (int i = 0; i < kModuleSize; ++i) {
      int out_channels = conv_arch_[i][1];
      vgg_seq_layers_->extend(
          *vgg_block(conv_arch_[i][0], in_channels, out_channels, false, in_shapes[i])
      );
      in_channels = out_channels;
    }
  }


  VGGNetwork_->extend(*vgg_seq_layers_);
  VGGNetwork_->extend(*classifier_);
  register_module("VGGNetwork_", VGGNetwork_);

}

#endif //PIMTORCH_VGG_H
