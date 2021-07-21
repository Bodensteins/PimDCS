#include <torch/torch.h>
#include <iostream>
#include "logic_array_interface.h"
#include "pim_conv.h"
#include "torch_conv.h"


using namespace torch;
using namespace std;
using namespace PIM;

const int64_t N = 2;
const int64_t Cin = 1;
const int64_t H = 7;
const int64_t W = 7;
const int64_t Cout = 1;
const int64_t kH = 4;
const int64_t kW = 2;
const int64_t s = 1;
const int64_t p = 4;


// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    conv1 = register_module("conv1",
        PimConv2d(
            ExpandingArray<4>({N, Cin, H, W}),
            PimArrayType::simple_logic_array,
            Conv2dOptions(Cin, Cout, {kH, kW}).stride(s).padding(p), true
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

  // const input
//  Tensor pim_input = torch::arange(N*Cin*H*W).reshape({N, H, W, Cin}).permute({0, 3, 1, 2})
//      .toType(torch::kFloat64).requires_grad_();
//  Tensor torch_input = torch::arange(N*Cin*H*W).reshape({N, H, W, Cin}).permute({0, 3, 1, 2})
//      .toType(torch::kFloat64).requires_grad_();
//  Tensor pim_weight = torch::arange(Cout*Cin*kH*kW).reshape({kH, kW, Cin, Cout}).permute({3, 2, 0, 1})
//      .toType(torch::kFloat64).requires_grad_();
//  Tensor torch_weight = torch::arange(Cout*Cin*kH*kW).reshape({kH, kW, Cin, Cout}).permute({3, 2, 0, 1})
//      .toType(torch::kFloat64).requires_grad_();
//  Tensor pim_bias = torch::ones({Cout}).toType(torch::kFloat64).requires_grad_();
//  Tensor torch_bias = torch::ones({Cout}).toType(torch::kFloat64).requires_grad_();

  // random input
  Tensor pim_input = torch::randn({N, Cin, H, W}).requires_grad_();
  Tensor torch_input = pim_input.clone().detach().requires_grad_();
  Tensor pim_weight = torch::randn({Cout, Cin, kH, kW}).requires_grad_();
  Tensor torch_weight = pim_weight.clone().detach().requires_grad_();
  Tensor pim_bias = torch::randn({Cout}).requires_grad_();
  Tensor torch_bias = pim_bias.clone().detach().requires_grad_();

  std::cout << "input" << std::endl;
  std::cout << torch_input << std::endl;
  std::cout << "============================" << std::endl;
  std::cout << "weight" << std::endl;
  std::cout << torch_weight << std::endl;
  std::cout << "============================" << std::endl;
  std::cout << "bias" << std::endl;
  std::cout << torch_bias << std::endl;
  std::cout << "============================" << std::endl;

  model.conv1->weight = pim_weight;
  model.conv1->bias = pim_bias;
  TorchConv2d torch_conv = TorchConv2d(
      Conv2dOptions(Cin, Cout, {kH, kW}).stride(s).padding(p).bias(true), torch_weight, torch_bias);

  auto pim_output = model.forward(pim_input);
  auto torch_output = torch_conv->forward(torch_input);

  pim_output.sum().backward();
  torch_output.sum().backward();

  auto t_weight_grad = torch_conv->weight.grad();
  auto t_bias_grad = torch_conv->bias.grad();
  auto t_input_grad = torch_input.grad();

  auto p_weight_grad = model.conv1->weight.grad();
  auto p_bias_grad = model.conv1->bias.grad();
  auto p_input_grad = pim_input.grad();

//  if (!torch::allclose(weight_grad, t_weight_grad, 1e-05, 1e-06)) {
    std::cout << "weight grad" << std::endl;
    Tensor w_err = p_weight_grad - t_weight_grad;
    std::cout << w_err.abs().max() << std::endl;
//    std::cout << t_weight_grad << std::endl;
    std::cout << "============================" << std::endl;
//  }
//  if (!torch::allclose(bias_grad, t_bias_grad, 1e-05, 1e-06)) {
    std::cout << "bias grad" << std::endl;
    Tensor b_err = p_bias_grad - t_bias_grad;
    std::cout << b_err.abs().max() << std::endl;
//    std::cout << t_bias_grad << std::endl;
    std::cout << "============================" << std::endl;
//  }
//  if (!torch::allclose(input_grad, t_input_grad, 1e-05, 1e-06)) {
    std::cout << "input grad" << std::endl;
    Tensor i_err = p_input_grad - t_input_grad;
    std::cout << i_err.abs().max() << std::endl;
//    std::cout << t_input_grad << std::endl;
    std::cout << "============================" << std::endl;
//  }


  return 0;
}