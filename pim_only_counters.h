/*******************************************************************************
* Copyright (c) 2020-2025
* Wuhan National Laboratory for Optoelectronics, Huazhong University of Science and Technology
* PI: Prof. Dan Feng & Prof. Wei Tong
* All rights reserved.
*   
* This source code is part of pimTorch: a circuit-algorithm framework for simulating neural 
* network in non-voltaile memory (e.g. ReRAM) with libtorch API (C++ API of Pytorch).
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list of this file: 
*   Bing Wu         Email: wubin200 at hust dot edu dot cn
*   Heng Zhou
*   Huan Cheng                  
********************************************************************************/
#ifndef PIMTORCH_PIM_ONLY_COUNTERS_H
#define PIMTORCH_PIM_ONLY_COUNTERS_H

#pragma once

#include "pim_array_example.h"
#include <cassert>
#include <mutex>
#include <algorithm>
#include <vector>

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;
using PIM::LogicArrayInterface;
using PIM::SimpleLogicArray;
using torch::TensorOptions;
using namespace PIM;

/*
*   This class manage allocation of phy array.
*   Logic array should call this class for allocating phy array.
*   So, we can add managment algorithm in this class for futuring work such as wear leveling.
*/
class phyArrayManager_counters
{
public:
    int allocPhyArray(int rowSize, int colSize, bool toGPU = false)
    {
        std::lock_guard<std::mutex> lk(mu);
        arrList.push_back(new phyArraySimple(rowSize, colSize, toGPU));
        return arrList.size() - 1;
    }

    phyArraySimple &access(int x)
    {
        return *arrList[x];
    }

    phyArraySimple &operator[](int x)
    {
        return *arrList[x];
    }

    void printAllInfo(std::ostream &os)
    {
        for (auto &i : arrList)
            i->print(os);
    }

    ~phyArrayManager_counters()
    {
        for (auto &i : arrList)
        {
            delete i;
        }
    }

private:
    std::vector<phyArraySimple *> arrList;
    static std::mutex mu;
};

std::mutex phyArrayManager_counters::mu;


pim_array_config decf_for_counters = {
    .rowSize = 256,
    .colSize = 256,
    .phyArrRowSize = 64,
    .phyArrColSize = 64,
    .inBits = 8,
    .outBits = 8,
    .unitBits = 8,
    .cellBits = 1,
    .has_negative_input = true,
    .max_phy_input_value = 128,
    .trunc_input = true,
    .dynamic_max_input = true
};


/*
*   only-counters in PIM array. 
*   tensor realMat is for computing
*   phyArray only counter its wear.
*/
class pimArrayExampleCounters : public LogicArrayInterface
{
public:
    pimArrayExampleCounters(int rowSizeIn, int colSizeIn, const torch::TensorOptions& op = {}, pim_array_config cf = decf_for_counters): LogicArrayInterface(rowSizeIn, colSizeIn)
    {
        cf.rowSize = rowSizeIn;
        cf.colSize = colSizeIn;
        init(cf, op);
    }

    pimArrayExampleCounters(const pim_array_config &cf, const torch::TensorOptions &op = {}): LogicArrayInterface(cf.rowSize, cf.colSize)
    {
        init(cf, op);
    }

    void init(const pim_array_config &cf, const torch::TensorOptions &op)
    {
        phyArrRowSize = cf.phyArrRowSize;
        phyArrColSize = cf.phyArrColSize;
        assert(cf.cellBits == 1);

        unitNumPerRow = phyArrColSize / cf.unitBits;
        assert(unitNumPerRow>0);
        usedcellsPerRow = unitNumPerRow * cf.unitBits;
        arrX_size = trunc_ceil((int)rowSize, phyArrRowSize);
        arrY_size = trunc_ceil((int)colSize, unitNumPerRow);

        inBits = cf.inBits;
        outBits = cf.outBits;
        unitBits = cf.unitBits;
        cellBits = cf.cellBits;
        unitLevels = 1 << unitBits;
        inLevels = 1 << inBits;
        outLevels = 1 << outBits;
        has_negative_input = cf.has_negative_input;
        max_phy_input_value = cf.max_phy_input_value;
        trunc_input = cf.trunc_input;
        dynamic_max_input = cf.dynamic_max_input;

        toGPU = op.device().is_cuda();
    
        realMat = torch::empty({rowSize, colSize}, torch::kFloat64).to(op.device());

        // std::cout << "realmat = " << realMat << std::endl;
        // std::cout << "in counter mat" << std::endl;
        device = toGPU? torch::kCUDA : torch::kCPU;
        //std::cout << "toGPU " << toGPU << std::endl;
        arr = std::vector<std::vector<int>>(arrX_size, std::vector<int>(arrY_size));
        for (int i = 0; i < arrX_size; ++i)
            for (int j = 0; j < arrY_size; ++j)
            {
                int k = phyArrMan.allocPhyArray(phyArrRowSize, phyArrColSize + 2, toGPU);
                arr[i][j] = k;
                for (int r = 0; r < phyArrRowSize; ++r)
                    phyArrMan[k].writeCell(r, phyArrColSize + 1, 1, torch::tensor({1}, TensorOptions(device)));
            }
        ImaxPCell = phyArrMan[0].ImaxPCell;
        IminPCell = phyArrMan[0].IminPCell;
        maxIsumPerPhyCol = ImaxPCell * phyArrRowSize;
        phyAllRowSize = phyArrRowSize*arrX_size;

        sizes_vec = std::vector<int64_t>({rowSize, colSize});


        // for (int i=0; i<phyAllRowSize; ++i)
        //     write_row(i, 0, torch::zeros({colSize}, torch::kFloat64));
    }

    void write_cell(int64_t row, int64_t col, const torch::Scalar &value) override;

    torch::Scalar read_cell(int64_t row, int64_t col) override;

    void write_row(int64_t row, int64_t col, const torch::Tensor &vec) override;

    torch::Tensor read_row(int64_t row, int64_t col, int64_t size) override; //从x行从y列开始读取size个列的数据

    torch::Tensor mv(const torch::Tensor &vec) override;

    torch::Tensor dot_column(const torch::Tensor &vec, int64_t col) override;

    torch::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) override; //数值乘以向量，需要输入操作的row号

    void write_mat(const torch::Tensor &mat) override;
    torch::Tensor read_mat() override;
    torch::Tensor mm(const torch::Tensor &mat) override;

    torch::IntArrayRef sizes() const override
    {
        return torch::IntArrayRef(sizes_vec);
    }

    torch::Tensor &resize(torch::IntArrayRef size) const override
    {
        std::cout << "currently, we do not support resize." << std::endl;
        throw("currently, we do not support resize.");
    }

    std::ostream &print(std::ostream &os) const override
    {
        for (int i = 0; i < arrX_size; ++i)
            for (int j = 0; j < arrY_size; ++j)
                phyArrMan[arr[i][j]].print(os);
    }

    at::Tensor input2digit(const at::Tensor &vec, double &max_one);

    void getRowPos(int row, int &X, int &arrRow)
    {
        X = row / phyArrRowSize;
        arrRow = row % phyArrRowSize;
    }

    void getColPos(int col, int &Y, int &arrCol)
    {
        Y = col / unitNumPerRow;
        arrCol = col % unitNumPerRow * unitBits;
    }

    void getPosition(int row, int col, int &X, int &Y, int &arrRow, int &arrCol)
    {
        arrRow = row % phyArrRowSize;
        arrCol = col % unitNumPerRow * unitBits;
        X = row / phyArrRowSize;
        Y = col / unitNumPerRow;
    }

    int unit2digit(double x)
    {
        return trunc_45((x + 1) / 2 * (unitLevels - 1));
    }

    torch::Tensor unit2digit(const torch::Tensor &x)
    {
        return x.add(1).div(2.0).mul(unitLevels - 1).add(0.5).to(torch::kInt32);
    }

    double digit2unit(int x)
    {
        return 1.0 * x / (unitLevels - 1) * 2 - 1;
    }

    torch::Tensor digit2unit(const torch::Tensor &x)
    {
        return x.to(torch::kFloat64).div(unitLevels-1).mul(2).subtract(1);
    }

private:
    static phyArrayManager_counters phyArrMan;
    std::vector<std::vector<int>> arr;

    int phyArrRowSize, phyArrColSize;
    int arrX_size, arrY_size;
    int unitNumPerRow;
    int usedcellsPerRow;

    bool has_negative_input;
    double max_phy_input_value;
    bool trunc_input;
    bool dynamic_max_input;
    int inBits, outBits, unitBits, cellBits;
    int unitLevels, inLevels, outLevels;
    double ImaxPCell, IminPCell, maxIsumPerPhyCol;
    int phyAllRowSize;
    bool toGPU;
    torch::DeviceType device;
    std::vector<int64_t> sizes_vec;
    torch::Tensor realMat;                  //in only counters mode,  we actually use realMat for computing. 
};

phyArrayManager_counters pimArrayExampleCounters::phyArrMan;

void pimArrayExampleCounters::write_cell(int64_t row, int64_t col, const torch::Scalar &value)
{
    int arrX, arrY, arrRowId, arrColId;

    getPosition(row, col, arrX, arrY, arrRowId, arrColId);
    int d = unit2digit(value.to<double>());
    at::Tensor data = torch::empty({unitBits}, TensorOptions(device).dtype(torch::kInt8));
    for (int i = 0; i < unitBits; ++i)
        data[i] = (d >> i) & 1;
    phyArrMan[arr[arrX][arrY]].writeCell(arrRowId, arrColId, unitBits, data);
    realMat[row][col] = digit2unit(d);
}

torch::Scalar pimArrayExampleCounters::read_cell(int64_t row, int64_t col)
{
    // int arrX, arrY, arrRowId, arrColId;

    // getPosition(row, col, arrX, arrY, arrRowId, arrColId);

    // at::Tensor data;
    // int x = 0;
    // phyArrMan[arr[arrX][arrY]].readCell(arrRowId, arrColId, unitBits, data);
    // for (int i = 0; i < unitBits; ++i)
    //     x = (data[i].item<int>() << i) | x;
    // return digit2unit(x);
    return realMat[row][col].item();
}

void pimArrayExampleCounters::write_row(int64_t row, int64_t col, const torch::Tensor &vec)
{
    int len = vec.size(0);
    int col_ed = col + len;

    int arrX, st_arrY, ed_arrY, arrRowId, st_arrColId, ed_arrColId;

    at::Tensor data = torch::empty({len * unitBits}, TensorOptions(device).dtype(torch::kInt8));
    at::parallel_for(0, len, 16, [&](int st, int ed)->void 
    {
        for (int i = st; i < ed; ++i)
        {
            int d = unit2digit(vec[i].item<double>());
            realMat[row][col+i] = digit2unit(d);
            for (int j = 0; j < unitBits; ++j)
                data[i * unitBits + j] = (d >> j) & 1;
        }
    });

    getRowPos(row, arrX, arrRowId);
    getColPos(col, st_arrY, st_arrColId);
    getColPos(col_ed, ed_arrY, ed_arrColId);

    if (st_arrY == ed_arrY)
    {
        phyArrMan[arr[arrX][st_arrY]].writeCell(arrRowId, st_arrColId, ed_arrColId - st_arrColId, data);
    }
    else
    {
        int ed = usedcellsPerRow - st_arrColId;

        phyArrMan[arr[arrX][st_arrY]].writeCell(arrRowId, st_arrColId, ed, data.slice(0, 0, ed));

        for (int y = st_arrY + 1; y < ed_arrY; ++y)
        {
            phyArrMan[arr[arrX][y]].writeCell(arrRowId, 0, usedcellsPerRow, data.slice(0, ed, ed + usedcellsPerRow));
            ed += usedcellsPerRow;
        }
        if (ed_arrColId != 0)
        {
            phyArrMan[arr[arrX][ed_arrY]].writeCell(arrRowId, 0, ed_arrColId, data.slice(0, ed, ed + ed_arrColId));
        }
    }
}

torch::Tensor pimArrayExampleCounters::read_row(int64_t row, int64_t col, int64_t size) //从x行从y列开始读取size个列的数据
{
    return realMat[row].slice(0, col, col+size);
}

at::Tensor pimArrayExampleCounters::input2digit(const at::Tensor &vec, double &max_one)
{
    if (dynamic_max_input)
    {
        if (has_negative_input) 
            max_one = std::min(max_phy_input_value, vec.abs().max().item<double>());
        else
            max_one = std::min(max_phy_input_value, vec.max().item<double>()); 
        //std::cout << "max_one = " << max_one << std::endl;
    }
    else
        max_one = max_phy_input_value;

    at::Tensor out;
    double k = (inLevels-1)/max_one;

    if (has_negative_input)
    {
        if (inBits<=1)
            std::cout << "has neg input, inbits should be >1. Operation undo" << std::endl;
        auto index_larger = vec>max_one;
        auto index_smaller = vec<-max_one;
        int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
        int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();

        if (!trunc_input && num_larger_than_max+num_smaller_than_min!=0)
            std::cout << "trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array." << std::endl;
        out = vec.add(max_one).mul(k/2).to(torch::kFloat64);
        out.index_put_({index_larger}, inLevels-1);
        out.index_put_({index_smaller}, 0);

        out = out.add(0.5).to(torch::kInt32).bitwise_xor((1 << (inBits -1)));
        max_one *= 2;
    }
    else
    {
        auto index_larger = vec>max_one;
        auto index_smaller = vec<0;
        int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>(); 
        int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();
        if (!trunc_input && num_larger_than_max+num_smaller_than_min!=0)
            std::cout << "trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array." << std::endl;
        out = vec.mul(k).add(0.5).to(torch::kInt32);
        out.index_put_({index_larger}, inLevels-1);
        out.index_put_({index_smaller}, 0);
    }
    
    if (out.dim()==1)
    {
        at::Tensor vecOut=torch::empty({inBits, out.size(0)}, TensorOptions(device).dtype(torch::kInt32));
        for (int i=0; i<inBits; ++i)
        {
            vecOut[i] = (out.bitwise_and(1 <<i)!=0);
        }
        return vecOut.t();
    }
    at::Tensor vecOut=torch::empty({out.size(1), out.size(0), inBits}, TensorOptions(device).dtype(torch::kInt32));
    out = out.t();
    for (int i=0; i<inBits; ++i)
    {
        vecOut.index_put_({"...", Slice(i, i+1)}, (out.bitwise_and(1 <<i)!=0).view({out.size(0), out.size(1), -1}));
    }
    return vecOut;
}

torch::Tensor pimArrayExampleCounters::mv(const torch::Tensor &vec)
{
    return torch::matmul(vec.reshape({1, -1}), realMat);
}

torch::Tensor pimArrayExampleCounters::dot_column(const torch::Tensor &vec, int64_t col)
{
    return torch::matmul(vec.reshape({1, -1}), realMat.index({Slice(), Slice(col, col+1)}));
}

torch::Tensor pimArrayExampleCounters::nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) //数值乘以向量，需要输入操作的row号
{   
    return realMat[row].slice(0, col, col+size).mul(value);
}

torch::Tensor pimArrayExampleCounters::mm(const torch::Tensor &mat) 
{
    if (mat.size(1)<realMat.size(0))
        return torch::matmul(mat, realMat.slice(0, 0, mat.size(1)));
    else
        return torch::matmul(mat, realMat);
}

/*
*   in face, now we only use write_mat in pim_linear & pim_conv.
*   so we optimizer it first. and then come to write_row & write_cell
*/
void pimArrayExampleCounters::write_mat(const torch::Tensor &mat) 
{
    int M = std::min(mat.size(0), rowSize);
    int N = std::min(mat.size(1), colSize);
    // at::parallel_for(0, len, 0, [&](int st, int ed)->void
    // {
    //     for (int i=st; i<ed; ++i)
    //         write_row(i, 0, mat[i]);
    // });
    torch::Tensor data = unit2digit(mat).index({Slice(0, M), Slice(0, N)});
    realMat.index_put_({Slice(0, M), Slice(0, N)}, digit2unit(data)); 
    torch::Tensor dataDigit = torch::empty({M, N * unitBits}, TensorOptions(device).dtype(torch::kInt8));

    for (int i=0; i<unitBits; ++i)
    {
        dataDigit.index_put_({Slice(), Slice(i, N*unitBits, unitBits)}, (data.bitwise_and(1<<i)!=0));
    }

    int ed_arrY, ed_arrColId;
    int ed_arrX, ed_arrRowId;
    getColPos(N, ed_arrY, ed_arrColId);
    getRowPos(M, ed_arrX, ed_arrRowId);
    
    at::parallel_for(0, ed_arrX*ed_arrY, 0, [&](int st, int ed)->void
    {
        for (int k=st; k<ed; ++k)
        {
            int i=k/ed_arrY;
            int j=k%ed_arrY;
            phyArrMan[arr[i][j]].writeMat(dataDigit.index({Slice(i*phyArrRowSize, (i+1)*phyArrRowSize), Slice(j*usedcellsPerRow, (j+1)*usedcellsPerRow)}),\
                            phyArrRowSize, phyArrColSize);
        }
    });
    // for (int i=0; i<ed_arrX; ++i)
    // {
    //     for (int j=0; j<ed_arrY; ++j)
    //     {
    //         phyArrMan[arr[i][j]].writeMat(dataDigit.index({Slice(i*phyArrRowSize, (i+1)*phyArrRowSize), Slice(j*usedcellsPerRow, (j+1)*usedcellsPerRow)}),\
    //                      phyArrRowSize, phyArrColSize);
    //     }
    // }
    if (ed_arrRowId!=0)
    {
        at::parallel_for(0, ed_arrY, 0, [&](int st, int ed)
        {
            for (int j=st; j<ed; ++j)
            {
                phyArrMan[arr[ed_arrX][j]].writeMat(dataDigit.index({Slice(ed_arrX*phyArrRowSize, M), Slice(j*usedcellsPerRow, (j+1)*usedcellsPerRow)}),\
                            ed_arrRowId, phyArrColSize);
            }
        });
        // for (int j=0; j<ed_arrY; ++j)
        // {
        //     phyArrMan[arr[ed_arrX][j]].writeMat(dataDigit.index({Slice(ed_arrX*phyArrRowSize, M), Slice(j*usedcellsPerRow, (j+1)*usedcellsPerRow)}),\
        //                  ed_arrRowId, phyArrColSize);
        // }
    }

    if (ed_arrColId!=0)
    {
        at::parallel_for(0, ed_arrX, 0, [&](int st, int ed)
        {
            for (int i=st; i<ed; ++i)
            {
                phyArrMan[arr[i][ed_arrY]].writeMat(dataDigit.index({Slice(i*phyArrRowSize, (i+1)*phyArrRowSize), Slice(ed_arrY*usedcellsPerRow, N*unitBits)}),\
                            phyArrRowSize, ed_arrColId);
            }
        });
        // for (int i=0; i<ed_arrX; ++i)
        // {
        //     phyArrMan[arr[i][ed_arrY]].writeMat(dataDigit.index({Slice(i*phyArrRowSize, (i+1)*phyArrRowSize), Slice(ed_arrY*usedcellsPerRow, N*unitBits)}),\
        //                  phyArrRowSize, ed_arrColId);
        // }
    }

    if (ed_arrColId!=0 && ed_arrColId!=0)
    {
        phyArrMan[arr[ed_arrX][ed_arrY]].writeMat(dataDigit.index({Slice(ed_arrX*phyArrRowSize, M), Slice(ed_arrY*usedcellsPerRow, N*unitBits)}),\
                         ed_arrRowId, ed_arrColId);
    }
}

torch::Tensor pimArrayExampleCounters::read_mat() 
{
    return realMat.clone();
}
#endif
