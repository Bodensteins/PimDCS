//
// Created by 15566 on 2021/7/19.
//
#include <torch/torch.h>
#include "pim_array_example.h"
#include "pim_array_config.h"
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

pair<int ,int> mvTest();
int getMaxPrecision(const at::Tensor &stdOut, const at::Tensor &myOut);
void printPrecision(int index);

int main()
{
    int testNum = 100;

    int errLowADC = 100;
    int errHighADC = 100;

    pair<int, int> err;
    for (int i = 0; i < testNum; ++i)
    {
        cout << "case " << i  << ':' << endl;
        err = mvTest();
        if (err.first < errLowADC)
        {
            errLowADC = err.first;
        }
        if (err.second < errHighADC)
        {
            errHighADC = err.second;
        }
    }

    cout << endl << "final result:" << endl;
    cout << "low adc precision: " << endl;
    printPrecision(errLowADC);
    cout << "high adc precision: " << endl;
    printPrecision(errHighADC);

    return 0;
}

pair<int ,int> mvTest()
{
    pim_array_pro_config config_low("../config/lowadc.yaml");
    pim_array_pro_config config_high("../config/highadc.yaml");

    const int M = 8;  // array of M x N
    const int N = 8;  //

    auto op = torch::TensorOptions(torch::kCPU).dtype(torch::kFloat64);

    pimArrayPro lowArray(M, N, op, &config_low);
    pimArrayPro highArray(M, N, op, &config_high);

    SimpleLogicArray softwareArray(M, N, op);
    at::Tensor v;

    auto writemat = at::randn({M, N}, torch::kFloat64);
    writemat = writemat.div(writemat.abs().max());

    softwareArray.write_mat(writemat);
    cout << "truearray:" << endl;
    cout << softwareArray.read_mat() << endl;
    lowArray.write_mat(writemat);
    cout << "lowarray:" << endl;
    cout << lowArray.read_mat() << endl;
    highArray.write_mat(writemat);
    cout << "higharray:" << endl;
    cout << highArray.read_mat() << endl;

    v = at::randn({1, M}, torch::kFloat64);
    v.div_(v.abs().max());

    //cout << v << endl;
    auto out = softwareArray.mm(v);

    //cout << "low:" << endl;
    auto lowout = lowArray.mm(v);

    //cout << "high:" << endl;
    auto highout = highArray.mm(v);

    out = out.squeeze();
    lowout = lowout.squeeze();
    highout = highout.squeeze();

    cout << "true out:" << endl << out  << endl << "low out:" << endl << lowout  << endl << "high out:" << endl << highout << endl;

    int err1 = getMaxPrecision(out, lowout);
    //cout << "out1 end" << endl << endl;
    int err2 = getMaxPrecision(out, highout);
    //cout << "low adc precision bit: " << err1 << " high adc precision bit: " << err2 << endl;

    return make_pair(err1, err2);
}

int getMaxPrecision(const at::Tensor &out, const at::Tensor &out1)
{
    auto delta = out - out1;
    double maxDistance  = delta.abs().max().item<double>();

    int n = 10;
    for (int i = 0; ; ++i, n *= 10)
    {
        if (n * maxDistance >= 1)
        {
            return i;
        }
    }
}

void printPrecision(int index)
{
    if (index < 0)
    {
        cout << "error! illegal precision!" << endl;
    }
    else if (index == 0)
    {
        cout << "1" << endl;
    }
    else
    {
        cout << "0.";
        for (int i = 0; i < index - 1; ++i)
        {
            cout << "0";
        }
        cout << "1" << endl;
    }
}

//
// Created by ubuntu on 5/11/21.
//

