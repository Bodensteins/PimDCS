#include <torch/torch.h>
#include "pure_array.h"
#include <ctime>
#include <omp.h>

using std::cout;
using std::endl;

// struct pim_array_config
// {
//     int32_t rowSize, colSize;
//     int32_t phyArrRowSize, phyArrColSize;
//     int32_t inBits, outBits, unitBits, cellBits;

//     bool has_negative_input;
//     double max_phy_input_value;
//     bool trunc_input;
//     bool dynamic_max_input;
// };

pim_array_config cf={
    .rowSize = 256,
    .colSize = 256,
    .phyArrRowSize = 64,
    .phyArrColSize = 64,
    .inBits = 8,
    .outBits = 8,
    .unitBits = 8,
    .cellBits = 1,
    .has_negative_input = true,
    .max_phy_input_value = 1,
    .trunc_input = false,
    .dynamic_max_input = true
};

const int N = 5;  // array of N x M
const int M = 5;   // 
const int len = 10;   // #len vectors

// we test a matrix of (len x N) mul (N x M)
void someSmallTest();
int main()
{
    someSmallTest();
    return 0;
    // torch::Tensor k = torch::ones({3, 4}).to(torch::kCUDA);
    // std::cout << k << std::endl;
    auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat64);
    pimArrayExampleCounters p(N, M, op, cf);
    SimpleLogicArray pp(N, M, op);

    torch::Tensor v=torch::ones({len, N}, torch::kFloat64);
    srand(time(0));
    for (int i=0; i<N; ++i)
    {
        for (int j=0; j<M; ++j)
        {
            double x= (rand()%1024-512)/512.0;
            pp.write_cell(i, j, x);
            p.write_cell(i, j, x);
        }
    }

    for (int i=0; i<len; ++i)
    {
        v[i] = torch::randn({N}, torch::kFloat64);
        v[i].div_(v[i].abs().max());
    }

    v = v.to(torch::kCUDA);
    // cout << "v=\n" << v << endl;
    auto st = clock();
    auto out = pp.mm(v);
    auto ed = clock();
    cout << "simpleLogic time = " <<(ed-st)/1.0/CLOCKS_PER_SEC << endl;
    
    st = clock();
    auto out1 = p.mm(v);


    ed = clock();
    cout << "our time = " <<(ed-st)/1.0/CLOCKS_PER_SEC << endl;
    // cout << "diff percent = \n" << (out-out1).div(out)*100 << endl;
    // cout << "real out=\n" << out <<endl;
    // cout << "our out1=\n" << out1 << endl;
    cout << (((out-out1).div(out)*100).abs()>=10).sum(0).sum(0).template item<double>()/len*100/M << "%" << endl;

    p.print(cout);
    return 0;
}

void someSmallTest()
{
    torch::Tensor tmp = torch::tensor({-0.5, 0.5, 0.2, 0.55});
    pimArrayExampleCounters p(5, 5, {}, cf);
    tmp = p.unit2digit(tmp);
    
    std::cout << tmp.reshape({1, -1}) << std::endl;

    tmp = p.digit2unit(tmp);

    std::cout << tmp.reshape({1, -1}) << std::endl;

    torch::Tensor a = torch::full({9, 9}, 1);
    at::parallel_for(0, 8, 0, [&](int st, int ed)->void
    {
        for (int i=st; i<ed; ++i)
            a[i][i] = i;
        printf("I am thread %d / %d \n",
            omp_get_thread_num(), omp_get_num_threads());
    });
    std::cout << a << std::endl;
    // torch::batch_norm_backward_reduce
}