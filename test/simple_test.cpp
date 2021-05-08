#include <torch/torch.h>
#include <iostream>
#include "ir_drop_solve.h"

using namespace torch;

int main() {
  Tensor w = torch::arange(3*2*3*3).reshape({3, 2, 3, 3}).toType(torch::kFloat32);
  auto flip_w = w.flip({2, 3});
  auto swap_flip_w = flip_w.permute({1, 0, 2, 3});
  auto kernel = swap_flip_w.reshape({swap_flip_w.size(0), -1}).t();
  std::cout << kernel << std::endl;

  at::Tensor t = torch::tensor({{1, 2}, {3, 4}});
  std::cout << t.gt(1).logical_not().logical_or(t.gt(2)) << std::endl;
  at::Tensor area = torch::zeros({5, 5}, torch::kBool).index_put_({Slice(2, 4), Slice(3, 4)}, true);
  std::cout << area << std::endl;
  return 0;
}
