//
// Created by chenghuan on 1/6/21.
//

#ifndef PIMTORCH_PHYARRAYSIMPLE_H
#define PIMTORCH_PHYARRAYSIMPLE_H

#include <torch/torch.h>
#include <torch/custom_class.h>

using torch::indexing::Slice;
using torch::TensorOptions;

/**
 * phyArraySimple implements a phy NVM crossbar array with size [rowSize, colSize]
 * The cell in array is digit. Only has minConduct and maxConduct two states.
 */
struct phyArraySimple
{
    phyArraySimple(int rowSize = 128, int colSize = 128, bool toGPU = false) : rowSize(rowSize), colSize(colSize)
    {
        double minConduct = 1e-6;   //define minimum & maximum conduct of a cell
        double maxConduct = 1e-4;
        double readVoltage = 0.5;   //define read voltage

        ImaxPCell = readVoltage * maxConduct;   //get current under read voltage
        IminPCell = readVoltage * minConduct;
        deltaI = ImaxPCell - IminPCell;
        totalCmpWrCnt = totalWrCnt = 0;

        device = toGPU? torch::kCUDA : torch::kCPU;
        torch::TensorOptions op(device);
        data = at::full({rowSize, colSize}, IminPCell, op.dtype(torch::kFloat64));   //data in type of current, Imin or Imax
        //dataDigit = at::full({rowSize, colSize}, 0, op.dtype(torch::kInt8));         //data in type of digit value 0(Imin)/1(Imax)
        dataDigit = at::zeros({rowSize, colSize}, op.dtype(torch::kInt8));//data in type of digit value 0(Imin)/1(Imax)
        cellWrCnt = at::zeros({rowSize, colSize}, op.dtype(torch::kInt64));
    }

    void writeCell(int row, int col, int len, const at::Tensor &in)
    {
        at::Tensor writeIn = at::full({len}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
        at::Tensor rowAdd = dataDigit[row];

        totalWrCnt += len;

        //rowAdd = rowAdd.index({torch::indexing::Slice(col, col + len)}).bitwise_xor(in);
        rowAdd = rowAdd.index({Slice(col, col + len)}).bitwise_xor(in);
        totalCmpWrCnt += rowAdd.sum().item<int64_t>();

//        cellWrCnt[row].index_put_({torch::indexing::Slice(col, col + len)},
//                                  cellWrCnt[row].index({torch::indexing::Slice(col, col + len)}).add(rowAdd));
        cellWrCnt[row].index({Slice(col, col + len)}).add_(rowAdd);

        writeIn.mul_(in).add_(IminPCell);
//        data[row].index_put_({torch::indexing::Slice(col, col + len)}, writeIn);
//        dataDigit[row].index_put_({torch::indexing::Slice(col, col + len)}, in);
        data[row].index_put_({Slice(col, col + len)}, writeIn);
        dataDigit[row].index_put_({Slice(col, col + len)}, in);
    }

    /*
    *   m -> in.size(0), n -> in.size(1)
    *   write mat in to our array.
    * */
    void writeMat(const at::Tensor &in, int m, int n)
    {
        at::Tensor writeIn = at::full({m, n}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
        at::Tensor add = dataDigit.index({Slice(0, m), Slice(0, n)}).bitwise_xor(in);

        totalWrCnt += (m*n);
//        totalCmpWrCnt += add.sum().item<int64_t>();
//        cellWrCnt.index_put_({Slice(0, m), Slice(0, n)}, cellWrCnt.index({Slice(0, m), Slice(0, n)}).add(add));
        cellWrCnt.index({Slice(0, m), Slice(0, n)}).add_(add);

        writeIn.mul_(in).add_(IminPCell);
        data.index_put_({Slice(0, m), Slice(0, n)}, writeIn);
        dataDigit.index_put_({Slice(0, m), Slice(0, n)}, in);
    }

    void readCell(int row, int col, int len, at::Tensor &out)
    {
        //out = dataDigit[row].index({torch::indexing::Slice(col, col + len)});
        out = dataDigit[row].index({Slice(col, col + len)});
    }

    void vmm(const at::Tensor &in, at::Tensor &out)
    {
        out = torch::matmul(data.t(), in);
    }

    void numMulVector(at::Scalar in, int row, int col, int len, at::Tensor &out)
    {
        //out = dataDigit[row].index({torch::indexing::Slice(col, col + len)}).mul(in);
        out = dataDigit[row].index({Slice(col, col + len)}).mul(in);
    }

    void print(std::ostream &os)
    {
        using std::endl;
        // os << "data = " << endl;
        // os << data << endl;
        os << "dataDigit = " << endl;
        os << dataDigit << endl;
        os << "cell write cnt = " << endl;
        os << cellWrCnt << endl;
        os << "totalWrCnt = " << totalWrCnt << endl;
        os << "totalCmpWrcnt = " << totalCmpWrCnt << endl;
        os << "--------------------" << endl;
    }
    at::Tensor data;
    at::Tensor dataDigit;
    at::Tensor cellWrCnt;
    int64_t totalWrCnt, totalCmpWrCnt;
    int rowSize, colSize;
    double ImaxPCell, IminPCell;
    double deltaI;
    torch::DeviceType device;
};

#endif //PIMTORCH_PHYARRAYSIMPLE_H
