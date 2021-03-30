#include <torch/torch.h>
#include "../pim_array_example.h"
#include "../pim_array_config.h"
#include <ctime>
#include <omp.h>

using std::cout;
using std::endl;

int M, N;
bool toGPU;
auto dev = torch::kCPU;

void writeTest();

int main()
{
    YAML::Node config = YAML::LoadFile("../unittest/config.yaml");
    M = config["M"].as<int>();
    N = config["N"].as<int>();
    toGPU = config["toGPU"].as<bool>();

    dev = toGPU? torch::kCUDA : torch::kCPU;
    writeTest();
    return 0;
}

void writeTest()
{
    pim_array_config cf;
    auto op = torch::TensorOptions(dev).dtype(torch::kFloat64);
    pimArrayPro pim(M, N, op, cf);
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
    pim.write_mat(tmp);
    simple.write_mat(tmp);
    
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {        
            cout << pim.read_cell(i, j).to<double>() << ' ' << simple.read_cell(i, j).to<double>() << endl;
        }
    }
    cout << (pim.read_mat()-simple.read_mat()).sum().item<double>()/M/N << endl;
}
