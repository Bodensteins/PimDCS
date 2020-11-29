//
// Created by 周恒 on 2020/11/23.
//

#include <torch/torch.h>

int main() {
  torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
  at::Tensor& b = a;
  b[0][0] = 3;
  std::cout << a << std::endl;
}