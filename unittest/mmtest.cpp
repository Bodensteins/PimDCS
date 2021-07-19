//
// Created by 15566 on 2021/7/19.
//
#include <torch/torch.h>
#include "pim_array_example.h"
#include "pim_array_config.h"
#include <map>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>

using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

pair<double ,double> mvTest(double &speedUp);
double get_err(const at::Tensor &std_out, const at::Tensor &out1);

int main()
{
    int testNum = 100;
    //int errnum = 0;
    double errmy = 0;
    double errtra = 0;
    double speedSum = 0;
    double speedUp;
    vector<double> spp;
    pair<double, double> err;
    for (int i = 0; i < testNum; ++i)
    {
        cout << "case " << i  << ':' << endl;
        err = mvTest(speedUp);
        errmy += err.first;
        errtra += err.second;
        spp.push_back(speedUp);
        //errnum += (err > 5);
    }
    std::sort(spp.begin(), spp.end());

    for (int i=10; i<testNum-10; ++i)
        speedSum += spp[i];

    cout << endl << "final result:" << endl;
    cout << "our average error rate: " << errmy/testNum << "%" << endl;
    cout << "traditional average error rate: " << errtra/testNum << "%" << endl;
    cout << "speed up: " << speedSum/(testNum-20) << endl;

    return 0;
}

pair<double ,double> mvTest(double &speedUp)
{
    pim_array_pro_config config("../config/5_2_4.yaml");
    pim_array_pro_config cfg_traditional("../config/5_2_4_traditional.yaml");
    double computeTime = config.latency.phyMMLatency * 1e-9;

    double softwareTime;
    double ourTime;
    //double speedUp;

    const int M = 512;  // array of M x N
    const int N = 512;  //

    auto op = torch::TensorOptions(torch::kCPU).dtype(torch::kFloat64);

    pimArrayPro myArray(M, N, op, &config);
    pimArrayPro traditionalArray(M, N, op, &cfg_traditional);

    SimpleLogicArray softwareArray(M, N, op);
    at::Tensor v;

    auto writemat = at::randn({M, N}, torch::kFloat64);
    writemat = writemat.div(writemat.abs().max());

    softwareArray.write_mat(writemat);
    myArray.write_mat(writemat);
//    cout << "myarray:" << endl;
//    myArray.print(cout);
    traditionalArray.write_mat(writemat);
//    cout << "traditional:" << endl;
//    traditionalArray.print(cout);

    v = at::randn({1, M}, torch::kFloat64);
    v.div_(v.abs().max());

    //cout << v << endl;

    auto st = clock();
    auto out = softwareArray.mm(v);
    auto ed = clock();
    softwareTime = (double)(ed - st) / CLOCKS_PER_SEC;
    cout << "software implemention time = " << softwareTime << endl;

    cout << "my:" << endl;
    auto out1 = myArray.mm(v);
    ourTime = config.inBits/config.inVBits * computeTime;
    cout << "our time = " << ourTime << endl;

    cout << "traditional:" << endl;
    auto out2 = traditionalArray.mm(v);

    speedUp = softwareTime / ourTime;
    cout << "speedup: " << speedUp << endl;

    out = out.squeeze();
    out1 = out1.squeeze();
    out2 = out2.squeeze();

    //cout << "out:" << endl << out << "out1:" << endl <<  out1 << "out2:" << endl << out2 << endl;

    double error = get_err(out, out1);
    //cout << "out1 end" << endl << endl;
    double err2 = get_err(out, out2);
    cout << "error rate: " << error << "%" << " traditional error rate: " << err2 << "%" << endl;

    return make_pair(error, err2);
}

double get_err(const at::Tensor &out, const at::Tensor &out1)
{
    double epsilon = 1e-3;
    auto newout = out.clone();

    for (int i = 0; i < out.size(0); ++i)
    {
        double outi = out[i].item<double>();
        double out1i = out1[i].item<double>();

        if (fabs(outi) < epsilon)
        {
            newout[i] = epsilon;
        }

        if (fabs(out1i - outi)/fabs(newout[i].item<double>()) > 10)
        {
            //cout << "out[" << i <<  "] = " << outi  << endl << "out1[" << i << "] = " << out1i << endl;
        }
    }

    return (out - out1).div(newout).abs().mean().item<double>() * 100;
}

double getMaxPrecision(const at::Tensor &out, const at::Tensor &out1)
{
    auto delta = out - out1;
    double maxDistance  = delta.abs().max().item<double>();

    int n = 1;
    for (int i = 0; ; ++i, n *= 10)
    {
        if (n * maxDistance >= 1)
        {
            return i;
        }
    }
}

//
// Created by ubuntu on 5/11/21.
//

