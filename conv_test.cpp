//
// Created by 周恒 on 2020/11/23.
//

#include <torch/torch.h>
#include <iostream>
#include "logic_array_interface.h"
#include "pim_conv.h"


using namespace torch;
using namespace std;
using namespace PIM;

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    conv1 = register_module("conv1",
        PimConv2D(ExpandingArray<4>({1, 2, 4, 4}),
            ExpandingArray<2>({3, 3}), 3, PimArrayType::simple_logic_array));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = conv1->forward(x);
    return x;
  }

  PimConv2D conv1{nullptr};
};

int main() {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  Tensor input = torch::arange(32).reshape({1, 2, 4, 4}).toType(torch::kFloat);
  auto output = model.forward(input);
  output.sum().backward();

  return 0;
}