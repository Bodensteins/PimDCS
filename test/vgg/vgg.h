//
// Created by 周恒 on 2021/2/6.
//

#ifndef PIMTORCH_VGG_H
#define PIMTORCH_VGG_H
#pragma once
#include <torch/torch.h>
#include <array>
#include <vector>
#include <dirent.h>
#include "pim_conv.h"
#include "pim_linear.h"

constexpr auto pim_mode = PIM::PimArrayType::pim_array_pro;
constexpr auto runDev = torch::kCUDA;
constexpr PIM::PIMRunMode fast_mode = PIM::PIMRunMode::inference;
constexpr auto ktype = torch::kF32;

torch::nn::Sequential vgg_block(const int kNumConvs, int in_channels, int out_channels, bool use_pim,
    std::vector<ExpandingArray<4>>& in_shapes) {
  torch::nn::Sequential vgg_layers;
  const int kKernelSize = 3;

  for (int i = 0; i < kNumConvs; ++i) {
    if (use_pim) {
      vgg_layers->push_back(PIM::PimConv2d(
          ExpandingArray<4>(in_shapes[i]),
          pim_mode,
          Conv2dOptions(in_channels, out_channels, kKernelSize).padding(1), fast_mode, TensorOptions(ktype).device(runDev)
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

  void load_params(const std::string& params_dir) {
    auto load_param_file = [](const std::string& path, Tensor& target) {
      std::vector<double> data;
      std::vector<int64_t> shape;
      unsigned long long size, ndim;
      double elem;
      std::ifstream x_ifs(path, std::ios::in | std::ios::binary);
      TORCH_CHECK(x_ifs.peek() != EOF, "parameter file is empty!");
      std::cout << "loading parameter: " << path << std::endl;
      x_ifs.read(reinterpret_cast<char *>(&size), sizeof(unsigned long long));
      x_ifs.read(reinterpret_cast<char *>(&ndim), sizeof(unsigned long long));
      shape.resize(ndim);
      for (unsigned long long i = 0; i < ndim; i++) {
        x_ifs.read(reinterpret_cast<char *>(&shape[i]), sizeof(unsigned long long));
      }
      data = std::vector<double>(size);
      for (unsigned long long i = 0; i < size; i++) {
        x_ifs.read(reinterpret_cast<char *>(&elem), sizeof(double));
        data[i] = elem;
      }
      x_ifs.close();

      ArrayRef<double> arrRef = ArrayRef<double>(data);
      Tensor tensor = torch::tensor(arrRef, torch::kDouble).reshape(IntArrayRef(shape));
      TORCH_CHECK(tensor.sizes() == target.sizes(), "the size of target tensor should be equal to loaded tensor.");
      target.set_data(tensor);
    };

    struct dirent *entry = nullptr;
    DIR *dp = nullptr;
    dp = opendir(params_dir.c_str());

    auto conv_params = this->vgg_seq_layers_->named_parameters();
    auto fc_params = this->classifier_->named_parameters();
    for (auto &e : conv_params) {
      std::string file_path = params_dir + "/features.module." + e.key() + ".bin";
      load_param_file(file_path, e.value());
    }
    for (auto &e : fc_params) {
      std::string file_path = params_dir + "/classifier." + e.key() + ".bin";
      load_param_file(file_path, e.value());
    }
    closedir(dp);
  }

public:
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
  int in_channels = in_shapes[0][0]->at(1);

  if (use_pim) {
    classifier_ = torch::nn::Sequential(
        /*The fully connected layer part*/
        torch::nn::Flatten(),
        // original: 512, 3 block: 2304
        torch::nn::Dropout(/*p=*/0.5),
        PIM::PimLinear(512, 512, batch_size, pim_mode, fast_mode, TensorOptions(ktype).device(runDev)),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.5),
        PIM::PimLinear(512, 512, batch_size, pim_mode, fast_mode, TensorOptions(ktype).device(runDev)),
        torch::nn::ReLU(),
        PIM::PimLinear(512, 10, batch_size, pim_mode, fast_mode, TensorOptions(ktype).device(runDev))
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
        torch::nn::Dropout(/*p=*/0.5),
        torch::nn::Linear(/*in_features=*/512, /*out_features=*/512),
        torch::nn::ReLU(),
        torch::nn::Dropout(/*p=*/0.5),
        torch::nn::Linear(/*in_features=*/512, /*out_features=*/512),
        torch::nn::ReLU(),
        torch::nn::Linear(/*in_features=*/512, /*out_features=*/10)
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
