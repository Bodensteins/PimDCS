#include <torch/torch.h>
#include <iostream>
#include "ir_drop_solve.h"

using namespace torch;
using std::endl;
using std::cout;

void test3x2()
{
    at::Tensor inV = torch::ones({1, 1, 3}, torch::kF64);
    at::Tensor G = torch::ones({3, 2}, torch::kF64)*1e-3;
    std::cout << ir_drop_solve_acc(inV, G, 3, 2, 1/2.0, 1/2.0) << std::endl;
    std::cout << ir_drop_solve_fast(inV, G, 3, 2, 5, 1/2.0, 1/2.0) << std::endl;
}

void test8x9()
{
    at::Tensor inV = torch::randn({1, 2, 8}, torch::kF64).abs();
    at::Tensor G = torch::randn({8, 9}, torch::kF64).abs()*1e-3;
    std::cout << ir_drop_solve_acc(inV, G, 8, 9, 1/2.8, 1/2.8) << std::endl;
    std::cout << ir_drop_solve_fast(inV, G, 8, 9, 5, 1/2.8, 1/2.8) << std::endl;
}

void test16x16()
{
    at::Tensor inV = torch::randn({1, 1, 16}, torch::kF64).abs();
    at::Tensor G = torch::randn({16, 16}, torch::kF64).abs()*1e-3;
    std::cout << torch::matmul(inV, G) << std::endl;
    std::cout << ir_drop_solve_acc(inV, G, 16, 16, 1/2.8, 1/2.8) << std::endl;
    std::cout << ir_drop_solve_fast(inV, G, 16, 16, 5, 1/2.8, 1/2.8) << std::endl;
}

void test32x32()
{

    at::Tensor inV = torch::ones({1, 1, 32}, torch::kF64);
    at::Tensor z = torch::zeros({1, 32}, torch::kF64);

    inV[0][0][0] = 0.5;
    inV[0][0][1] = 3;
    at::Tensor G = torch::ones({32, 32}, torch::kF64)*1e-4;
    G.index({Slice(0, 1), Slice()}) = 1e-6;

    //std::cout << G << std::endl;
    std::cout << torch::matmul(inV, G) << std::endl;
    std::cout << ir_drop_solve_acc(inV, G, 32, 32, 1/2.8, 1/2.8) << std::endl;
    std::cout << ir_drop_solve_fast(inV, G, 32, 32, 10, 1/2.8, 1/2.8) << std::endl;
}

void test64x64()
{

    at::Tensor inV = torch::ones({1, 1, 64}, torch::kF64);
    at::Tensor z = torch::zeros({1, 64}, torch::kF64);

    inV[0][0][0] = 0.5;
    inV[0][0][1] = 3;
    at::Tensor G = torch::ones({64, 64}, torch::kF64)*1e-4;
    G.index({Slice(0, 1), Slice()}) = 1e-6;

    //std::cout << G << std::endl;
    std::cout << torch::matmul(inV, G) << std::endl;
    std::cout << ir_drop_solve_acc(inV, G, 64, 64, 1/2.8, 1/2.8) << std::endl;
    std::cout << ir_drop_solve_fast(inV, G, 64, 64, 10, 1/2.8, 1/2.8) << std::endl;
}

int main()
{
    test3x2();
    test64x64();
    if (ir_drop_fastMode_check(10, 0.3))
    {
        std::cout << "pass check" << std::endl;
    }
    else
    {
        cout <<"not pass check" << endl;
    }
    
    return 0;
}