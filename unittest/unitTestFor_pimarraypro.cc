#include <torch/torch.h>
#include "pim_array_example.h"
#include "pim_array_config.h"
#include <ctime>
#include <omp.h>

using std::cout;
using std::endl;

int M, N;
bool toGPU;
auto dev = torch::kCPU;

void writeTest();
void preWorkTest();
void mmTest();
int main()
{
    YAML::Node config = YAML::LoadFile("../unittest/config.yaml");
    M = config["M"].as<int>();
    N = config["N"].as<int>();
    toGPU = config["toGPU"].as<bool>();

    dev = toGPU? torch::kCUDA : torch::kCPU;
    //writeTest();

    //preWorkTest();
    mmTest();
    return 0;
}

void writeTest()
{
    pim_array_config cf;
    auto op = torch::TensorOptions(dev).dtype(torch::kFloat64);
    pimArrayPro pim(M+1, N+1, op);
    SimpleLogicArray simple(M, N, op);
    srand(time(NULL));
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         double x = (rand() % 1024 - 512) / 512.0;
    //         pim.write_cell(i, j, x);
    //         simple.write_cell(i, j, x);
        
    //         cout << pim.read_cell(i, j).to<double>() << ' ' << simple.read_cell(i, j).to<double>() << endl;
    //     }
    // }

    at::Tensor tmp = torch::rand({M, N}, op)*-5;
    //cout << tmp << endl;
    cout << "write" << endl;
    pim.write_mat(tmp, 1, 1);
    simple.write_mat(tmp);
    
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {        
            cout << pim.read_cell(i+1, j+1).to<double>() << ' ' << simple.read_cell(i, j).to<double>() << endl;
        }
    }
    cout << (pim.read_mat(1, 1)-simple.read_mat()).sum().item<double>()/M/N << endl;
}

void preWorkTest()
{
    pim_array_pro_config cf;
    double max_one;
    at::Tensor mat = torch::tensor({{1, 3}, {-5, 7}});
    cout << phyArrayPro::preWorkForMM(mat, &cf, max_one) << endl;
}

void mmTest()
{
    pim_array_pro_config cf;    
    at::Tensor mat = torch::rand({2, 2}, torch::kF64)*-4;

    pimArrayPro test(2, 2);
    auto data = torch::rand({2, 2}, torch::kF64)*-1;
    cout << "true=" << matmul(mat, data) << endl;
    test.write_mat(data);
    cout << test.read_mat() << endl;
    cout << test.mm(mat) << endl;
    

}