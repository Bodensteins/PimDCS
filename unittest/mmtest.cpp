//
// Created by 15566 on 2021/7/19.
//
#include <torch/torch.h>
#include "pim_array_example.h"
#include "pim_array_config.h"
#include <vector>
#include <algorithm>

using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

double mvTest();
double getMaxPrecision(const at::Tensor &stdOut, const at::Tensor &myOut);

int main()
{
    int testNum = 1000;

    double errMax = 0;

    double err;
    for (int i = 0; i < testNum; ++i)
    {
        err = mvTest();
        if (err > errMax)
        {
            errMax = err;
        }
    }

    cout << endl << "final result:" << endl;
    cout << "max err: " << errMax << endl;
    //printPrecision(errHighADC);

    return 0;
}

double mvTest()
{
    const int M = 64;  // array of M x N
    const int N = 8;  //

    auto op = torch::TensorOptions(torch::kCPU).dtype(torch::kFloat64);

    pimArrayPro crossbarArray(M, N, op, &pro_decf());

    SimpleLogicArray softwareArray(M, N, op);
    at::Tensor v;

    auto writemat = at::randn({M, N}, torch::kFloat64);
    writemat = writemat.div(writemat.abs().max());

    softwareArray.write_mat(writemat);
//    cout << "truearray:" << endl;
//    cout << softwareArray.read_mat() << endl;
    crossbarArray.write_mat(writemat);
//    cout << "crossbarArray:" << endl;
//    cout << crossbarArray.read_mat() << endl;

    v = at::randn({1, M}, torch::kFloat64);
    v.div_(v.abs().max());
    //cout << v << endl;
    auto out = softwareArray.mm(v);

    auto crossbarArrayOut = crossbarArray.mm(v);

    out = out.squeeze();
    crossbarArrayOut = crossbarArrayOut.squeeze();

    //cout << "true out:" << endl << out  << endl << "crossbarArrayOut out:" << endl << crossbarArrayOut << endl;

    double err = getMaxPrecision(out, crossbarArrayOut);
    //cout << "err: " << err << endl;

    return err;
}

double getMaxPrecision(const at::Tensor &stdOut, const at::Tensor &myOut)
{
    auto delta = stdOut - myOut;
    double maxDistance  = delta.abs().max().item<double>();

    return maxDistance;
}

