#include <iostream>
#include <torch/torch.h>
#include "logic_array_interface.h"

int main() {
  SimpleLogicArray arr = SimpleLogicArray(10, 10, torch::kInt32);
  for (int row = 0; row < 10; row++) {
    for (int col = 0; col < 10; col++) {
      arr.write_cell(row, col, torch::Scalar(row * 10 + col));
    }
  }
  std::cout << arr << std::endl;
  std::cout << arr.mv(torch::full({10}, 2, torch::kInt32)) << std::endl;
  arr.write_row(0, 0, torch::ones({3}, torch::kInt32));
  std::cout << arr << std::endl;
  std::cout << arr.read_row(2, 2, 3) << std::endl;
  std::cout << arr.nmv(3, 4, torch::Scalar(3), 2) << std::endl;
  torch::Tensor t = torch::ones({2, 2});
  t.resize_({3, 3});
  std::cout << t << std::endl;
  return 0;
}
