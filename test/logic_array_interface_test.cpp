//
// Created by 周恒 on 2020/11/23.
//

#include <iostream>
#include <torch/torch.h>
#include "../include/pim_linear.h"

using namespace torch::nn;
using namespace PIM;

struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", PimLinear(6, 7, 1, PimArrayType::simple_logic_array));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = fc1->forward(x);
    return x;
  }

  PimLinear fc1{nullptr};
};

int main() {
//  Net model;
//  model.to(torch::kCPU);
//  model.train();
//
//  torch::Tensor input = torch::rand({1, 6}).requires_grad_();
//  auto output = model.forward(input);
//  output.backward({torch::ones({1, 7})});

  namespace F = torch::nn::functional;
  torch::Tensor inp = torch::randn({1, 2, 4, 4}).requires_grad_();
  torch::Tensor w = torch::randn({2, 2, 3, 3}).requires_grad_();
  torch::Tensor b = torch::randn({2}).requires_grad_();
  auto inp_unf = F::unfold(inp, F::UnfoldFuncOptions({3, 3}));
  auto out_unf = inp_unf.transpose(1, 2).matmul(w.view({w.size(0), -1}).t()).transpose(1, 2);
  auto out = F::fold(out_unf, F::FoldFuncOptions({{2, 2}, {1, 1}}));
  inp.print();
  inp_unf.print();
  w.view({w.size(0), -1}).print();
  out_unf.print();

  out[0][0].add_(b[0]);out[0][1].add_(b[1]);

  std::cout << (F::conv2d(inp, w, F::Conv2dFuncOptions().bias(b)) - out).abs().max() << std::endl;

  // ======= backward




  // ===========
  std::vector<torch::Tensor> l;
  SimpleLogicArray pim_w = SimpleLogicArray(9, 2);
  pim_w.write_mat(w.view({w.size(0), -1}).t());

  for (int i = 0; i < inp_unf.size(0); i++) {
    l.push_back(pim_w.mm(inp_unf.transpose(1, 2)[i]));
  }
  auto pim_out = F::fold(torch::stack(l, 0).transpose(1, 2), F::FoldFuncOptions({{2, 2}, {1, 1}}));
  pim_out[0][0].add_(b[0]);pim_out[0][1].add_(b[1]);
  std::cout << (F::conv2d(inp, w, F::Conv2dFuncOptions().bias(b)) - pim_out).abs().max() << std::endl;

  out.backward(torch::ones({1, 2, 2, 2}));
  std::cout << "========== weight ==========" << std::endl;
  std::cout << w.grad() << std::endl;
  std::cout << "========== bias ==========" << std::endl;
  std::cout << b.grad() << std::endl;
  std::cout << "========== input ==========" << std::endl;
  std::cout << inp.grad() << std::endl;

  return 0;
}