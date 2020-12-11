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

const int64_t N = 2;
const int64_t Cin = 2;
const int64_t H = 32;
const int64_t W = 32;
const int64_t Cout = 3;

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
//    conv1 = register_module("conv1",
//        PimConv2d(ExpandingArray<4>({2, 2, 4, 4}),
//            ExpandingArray<2>({3, 3}), 3, PimArrayType::simple_logic_array));
    conv1 = register_module("conv1",
        PimConv2d(
            ExpandingArray<4>({N, Cin, H, W}),
            PimArrayType::simple_logic_array,
            Conv2dOptions(Cin, Cout, {5, 5}).stride(1).padding(0)
            ));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = conv1->forward(x);
    return x;
  }

  PimConv2d conv1{nullptr};
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
  model.to(device, torch::kFloat64);

  Conv2d torch_conv = torch::nn::Conv2d(
      Conv2dOptions(Cin, Cout, 5).stride(1).padding(0));
  torch_conv->weight = model.conv1->weight;
  torch_conv->bias = model.conv1->bias;

  Tensor input = torch::arange(N*Cin*H*W).reshape({N, Cin, H, W}).toType(torch::kFloat64).requires_grad_();
  Tensor torch_input = torch::arange(N*Cin*H*W).reshape({N, Cin, H, W}).toType(torch::kFloat64).requires_grad_();

  auto output = model.forward(input);
  auto torch_output = torch_conv->forward(torch_input);

  output.sum().backward();
  torch_output.sum().backward();

  auto t_weight_grad = torch_conv->weight.grad();
  auto t_bias_grad = torch_conv->bias.grad();
  auto t_input_grad = torch_input.grad();

  auto weight_grad = model.conv1->weight.grad();
  auto bias_grad = model.conv1->bias.grad();
  auto input_grad = input.grad();

//  if (!torch::allclose(weight_grad, t_weight_grad, 1e-05, 1e-06)) {
    std::cout << "weight grad" << std::endl;
    Tensor w_err = weight_grad - t_weight_grad;
    std::cout << w_err.abs().max() << std::endl;
//  }
//  if (!torch::allclose(bias_grad, t_bias_grad, 1e-05, 1e-06)) {
    std::cout << "bias grad" << std::endl;
    Tensor b_err = bias_grad - t_bias_grad;
    std::cout << b_err.abs().max() << std::endl;
//  }
//  if (!torch::allclose(input_grad, t_input_grad, 1e-05, 1e-06)) {
    std::cout << "input grad" << std::endl;
    Tensor i_err = input_grad - t_input_grad;
    std::cout << i_err.abs().max() << std::endl;
//  }

  return 0;
}