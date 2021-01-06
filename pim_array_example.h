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
#ifndef PIMTORCH_ARRAY_EXAMPLE_H
#define PIMTORCH_ARRAY_EXAMPLE_H

#pragma once

#include "logic_array_interface.h"
#include <cassert>
#include <mutex>
#include <algorithm>
#include <vector>
#include "phyArraySimple.h"
#include "pim_array_config.h"

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;
using PIM::LogicArrayInterface;
using PIM::SimpleLogicArray;
using torch::TensorOptions;
using namespace PIM;

template <typename T>
inline T trunc_ceil(T x, T mod)
{
    return (x + mod - 1) / mod;
}

inline int64_t trunc_45(double x)
{
    return (int64_t)(x + 0.5);
}




/*
*   This class manage allocation of phy array.
*   Logic array should call this class for allocating phy array.
*   So, we can add managment algorithm in this class for futuring work such as wear leveling.
*/
class phyArrayManager
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

    ~phyArrayManager()
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

//std::mutex phyArrayManager::mu;


class pimArrayExample : public LogicArrayInterface
{
public:
//    pimArrayExample(int rowSizeIn, int colSizeIn, const torch::TensorOptions& op = {}, pim_array_config cf = decf): LogicArrayInterface(rowSizeIn, colSizeIn)
//    {
//        cf.rowSize = rowSizeIn;
//        cf.colSize = colSizeIn;
//        init(cf, op);
//    }

    pimArrayExample(const pim_array_config &cf, const torch::TensorOptions &op = {}): LogicArrayInterface(cf.rowSize, cf.colSize)
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
    static phyArrayManager phyArrMan;
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
};

//phyArrayManager pimArrayExample::phyArrMan;


#endif
