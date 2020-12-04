//
// Created by 周恒 on 2020/11/23.
//

#include <torch/torch.h>
#include <iostream>
#include "logic_array_interface.h"

using namespace torch;
using namespace std;
using namespace PIM;

int main() {
  ExpandingArray<4> input_shape({1, 2, 3, 4});
  std::cout << input_shape << std::endl;
  return 0;
}