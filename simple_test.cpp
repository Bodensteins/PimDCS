//
// Created by 周恒 on 2020/12/11.
//

#include <torch/torch.h>
#include <iostream>

using namespace torch;

using std::cout;
using std::endl;

int main() {
  Tensor w = torch::arange(3*2*3*3).reshape({3, 2, 3, 3}).toType(torch::kFloat32);
  auto flip_w = w.flip({2, 3});
  auto swap_flip_w = flip_w.permute({1, 0, 2, 3});
  auto kernel = swap_flip_w.reshape({swap_flip_w.size(0), -1}).t();
  std::cout << kernel << std::endl;
  Tensor k = torch::tensor({{2.0, 3.0}, {1.4, 0.7}});
  cout << (k>1) << endl;
  k.index_put_({k>1}, 1.0);
  cout << k << endl;
  return 0;
}
