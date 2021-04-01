#include <torch/torch.h>
#include "pim_utils.h"
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

pim_array_config cf;


const int N = 5;  // array of N x M
const int M = 5;   // 
const int len = 10;   // #len vectors

// we test a matrix of (len x N) mul (N x M)
void someSmallTest();
void accuracyTest();
void scheduleTest();

int main()
{
    //someSmallTest();
    //accuracyTest();
    scheduleTest();
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
    cout << "v=\n" << v << endl;
    auto st = clock();
    auto out = pp.mm(v);
    auto ed = clock();
    cout << "simpleLogic time = " <<(ed-st)/1.0/CLOCKS_PER_SEC << endl;
    
    st = clock();
    auto out1 = p.mm(v);


    ed = clock();
    cout << "our time = " <<(ed-st)/1.0/CLOCKS_PER_SEC << endl;
    cout << "diff percent = \n" << (out-out1).div(out)*100 << endl;
    //cout << "real out=\n" << out <<endl;
    //cout << "our out1=\n" << out1 << endl;
    cout << (((out-out1).div(out)*100).abs()>=10).sum(0).sum(0).template item<double>()/len*100/M << "%" << endl;

    //p.print(cout);
    return 0;
}

void accuracyTest()
{
    auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat64);
    pimArrayExampleCounters p(N, M, op, cf);
    torch::Tensor pp = torch::rand({N, M}).to(op);


    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);
    p.phyArrMan.schedule(1, 2, 0);

    p.write_mat(pp);


    auto k = p.read_mat().to(op);

    cout << k << endl;
    cout << pp << endl;
    cout << (k-pp).div(pp)*100 << endl;


    phyArraySimpleEx a(4, 6, false);
    //a.writeMat(torch::tensor({{1, 0}, {0, 1}}), 2, 2);
    a.rotate();
    a.rotate();
    //a.writeMat(torch::tensor({{1, 1}, {0, 1}}), 2, 2);
    a.rotate();
    //a.writeMat(torch::tensor({{1, 1}, {0, 1}}), 2, 2);
    for (int i=0; i<3; ++i)
        a.rotate();
    //a.writeMat(torch::tensor({{0, 1}, {0, 1}}), 2, 2);

    for (int i=0; i<9; ++i)
       a.rotate(); 

    a.writeMat(torch::tensor({{1, 1, 1, 0}, {1, 1, 0, 1}, {1, 0, 1, 1}, {0, 1, 1, 1}}), 4, 4);
    a.writeCell(0, 5, 1, torch::tensor({1}));
    a.print(cout);
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

void scheduleTest()
{
    auto op = torch::TensorOptions(torch::kCPU).dtype(torch::kFloat64);
    pimArrayExampleCounters pim(128, 24, op, cf);
    torch::Tensor pp1 = torch::cat({torch::ones({64, 8}).to(op), torch::ones({64, 8}).to(op).mul(-1), torch::ones({64, 8}).to(op).mul(-1)}, 1);
    torch::Tensor pp2 = torch::ones({64, 24}).to(op).mul(-1);

    pp1 = torch::cat({pp1, pp1}, 0);
    pp2 = torch::cat({pp2, pp2}, 0); 

    for (int i=0; i<64*64*1; ++i)
    {
        pim.write_mat(pp1);
        pim.write_mat(pp2);
        pim.phyArrMan.schedule(1, 64*64, 2);
    }

    pim.print(cout);
    cout << "out" << endl;
}
