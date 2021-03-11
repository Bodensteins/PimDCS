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
#include <map>

using std::swap;
using std::pair;
using std::make_pair;
using std::vector;
using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;
using PIM::LogicArrayInterface;
using PIM::SimpleLogicArray;
using torch::TensorOptions;
using namespace PIM;

struct phyArraySimpleEx
{
    phyArraySimpleEx(int rowSize = 128, int colSize = 128, bool toGPU = false) : rowSize(rowSize), colSize(colSize)
    {
        double minConduct = 1e-6;   //define minimum & maximum conduct of a cell
        double maxConduct = 1e-4;
        double readVoltage = 0.5;   //define read voltage

        ImaxPCell = readVoltage * maxConduct;   //get current under read voltage
        IminPCell = readVoltage * minConduct;
        deltaI = ImaxPCell - IminPCell;
        totalIntervalWrCnt = totalCmpWrCnt = totalWrCnt = 0;

        device = toGPU? torch::kCUDA : torch::kCPU;
        torch::TensorOptions op(device);
        data = at::full({rowSize, colSize}, IminPCell, op.dtype(torch::kFloat64));   //data in type of current, Imin or Imax
        dataDigit = at::full({rowSize, colSize}, 0, op.dtype(torch::kInt8));         //data in type of digit value 0(Imin)/1(Imax)
        cellWrCnt = at::zeros({rowSize, colSize}, op.dtype(torch::kInt64));                 

        rightRotate = downRotate = 0;
        colSize_ = colSize-2;
    }

    void writeCell(int row, int col, int len, const at::Tensor &in)
    {
        auto write_it = [&](int row, int col, int len, const at::Tensor &in) -> void
        {
            at::Tensor writeIn = at::full({len}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
            at::Tensor rowAdd = dataDigit[row];

            totalWrCnt += len;              // write count 
            int64_t tmp;
            rowAdd = rowAdd.index({torch::indexing::Slice(col, col + len)}).bitwise_xor(in);
            totalCmpWrCnt += (tmp = rowAdd.sum().item<int64_t>());  //cmp write count, only write different data to origin data will be counted
            totalIntervalWrCnt += tmp;
            cellWrCnt[row].index_put_({torch::indexing::Slice(col, col + len)},
                    cellWrCnt[row].index({torch::indexing::Slice(col, col + len)}).add(rowAdd)); //cmp write count of each cell 

            writeIn.mul_(in).add_(IminPCell);
            data[row].index_put_({torch::indexing::Slice(col, col + len)}, writeIn);    //data in type of current
            dataDigit[row].index_put_({torch::indexing::Slice(col, col + len)}, in);    //data in type of digit
        };
        
        /* ref colum don't rotate, rotate only happen to data region */
        if (col == colSize-1 && len == 1) // write ref column 1, never rotate
        {
            write_it(row, col, len, in);
            return;
        }

        row = (row+downRotate)%rowSize;
        col = (col+rightRotate)%colSize_;

        if (col+len<=colSize_)
        {
            write_it(row, col, len, in);
        }
        else
        {
            int len1, len2;
            len1 = colSize_ - col;
            len2 = len-len1; 
            write_it(row, col, len1, in.index({Slice(0, len1)}));
            write_it(row, 0, len2, in.index({Slice(len1, len1+len2)}));
        }


    }

    void rotate()
    {
        ++rightRotate;
        if (rightRotate == colSize_)
        {
            ++downRotate;
            rightRotate = 0;
            if (downRotate == rowSize)
                downRotate = 0;
        }
    }
    
    void clearIntervalCount()
    {
        totalIntervalWrCnt = 0;
    }

    void initWriteMat(const at::Tensor &in)
    {
        auto write_it = [&](const at::Tensor &in, int x, int y, int m, int n) -> void
        {
            //std::cout << in << std::endl;
            at::Tensor writeIn = at::full({m, n}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
            //std::cout << dataDigit.index({Slice(x, x+m), Slice(y, y+n)}) << std::endl;
            at::Tensor add = dataDigit.index({Slice(x, x+m), Slice(y, y+n)}).bitwise_xor(in.to(torch::kInt8));

            totalWrCnt += (m*n);
            int64_t tmp;
            totalCmpWrCnt += (tmp = add.sum().item<int64_t>());
            totalIntervalWrCnt += tmp;
            cellWrCnt.index_put_({Slice(x, x+m), Slice(y, y+n)}, cellWrCnt.index({Slice(x, x+m), Slice(y, y+n)}).add(add));

            writeIn.mul_(in).add_(IminPCell);
            data.index_put_({Slice(x, x+m), Slice(y, y+n)}, writeIn);
            dataDigit.index_put_({Slice(x, x+m), Slice(y, y+n)}, in);
        };
        write_it(in, 0, 0, rowSize, colSize);
    }
    /*
    *   m -> in.size(0), n -> in.size(1)
    *   write mat into our array.
    *   similar to writeCell
    * */
    void writeMat(const at::Tensor &vec, int m, int n)
    {
        auto write_it = [&](const at::Tensor &in, int x, int y, int m, int n) -> void
        {
            //std::cout << in << std::endl;
            at::Tensor writeIn = at::full({m, n}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
            //std::cout << dataDigit.index({Slice(x, x+m), Slice(y, y+n)}) << std::endl;
            at::Tensor add = dataDigit.index({Slice(x, x+m), Slice(y, y+n)}).bitwise_xor(in.to(torch::kInt8));

            totalWrCnt += (m*n);
            int64_t tmp;
            totalCmpWrCnt += (tmp = add.sum().item<int64_t>());
            totalIntervalWrCnt += tmp;
            cellWrCnt.index_put_({Slice(x, x+m), Slice(y, y+n)}, cellWrCnt.index({Slice(x, x+m), Slice(y, y+n)}).add(add));

            writeIn.mul_(in).add_(IminPCell);
            data.index_put_({Slice(x, x+m), Slice(y, y+n)}, writeIn);
            dataDigit.index_put_({Slice(x, x+m), Slice(y, y+n)}, in);
        };

        int stx, sty, edx, edy;
        stx = downRotate;
        sty = rightRotate;

        edx = (stx+m-1)%rowSize;
        edy = (sty+n-1)%colSize_;

        if (edx>=stx)
        {
            if (edy>=sty)
            {
                write_it(vec, stx, sty, m, n);
            }
            else
            {
                //std::cout << "x y m n = " << stx << ' ' << sty << ' ' << m << ' ' << colSize_-sty<< std::endl;
                //std::cout << "x y m n = " << stx << ' ' << 0 << ' ' << m << ' ' << edy+1<< std::endl;
                write_it(vec.index({Slice(), Slice(0, colSize_-sty)}), stx, sty, m, colSize_-sty);
                write_it(vec.index({Slice(), Slice(colSize_-sty)}), stx, 0, m, edy+1);
            }
        }
        else
        {
            if (edy>=sty)
            {
                write_it(vec.index({Slice(0, rowSize-stx)}), stx, sty, rowSize-stx, n);
                write_it(vec.index({Slice(rowSize-stx)}), 0, sty, edx+1, n);
            }
            else
            {
                write_it(vec.index({Slice(0, rowSize-stx), Slice(0, colSize_-sty)}), stx, sty, rowSize-stx, colSize_-sty);
                write_it(vec.index({Slice(0, rowSize-stx), Slice(colSize_-sty)}), stx, 0, rowSize-stx, edy+1);
                write_it(vec.index({Slice(rowSize-stx), Slice(0, colSize_-sty)}), 0, sty, edx+1, colSize_-sty);
                write_it(vec.index({Slice(rowSize-stx), Slice(colSize_-sty)}), 0, 0, edx+1, edy+1);
            }
        }
    }

    /*
     *
     * readCell from [row, col] to [row, col+len-1]
     * As we have stored data in type of digit in dataDigit, so just return it.
     *
     *
     * */
    void readCell(int row, int col, int len, at::Tensor &out)
    {
        row = (row+downRotate)%rowSize;
        col = (col+rightRotate)%colSize_;

        if (col+len<=colSize)
            out = dataDigit[row].index({torch::indexing::Slice(col, col + len)});
        else
        {
            int len1 = colSize_ - col;
            int len2 = len-len1;
            out = torch::cat({dataDigit[row].index({Slice(col, col+len1)}), dataDigit[row].index({Slice(0, len2)})});
        }
    }

    /*
     * VMM 
     * implement   out = data^t * in.    ^t means transpose
     *
     * */
    void vmm(const at::Tensor &in, at::Tensor &out)
    {
        /* only counter mode, don't use vmm */
    }

    /*
     *  number mul vector
     *  out = in * data[row][col <---> col+len-1]
     *
     * */
    void numMulVector(at::Scalar in, int row, int col, int len, at::Tensor &out)
    {
        /* only counter mode, don't use nmv*/
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
        os << "totalCmpWrCnt = " << totalCmpWrCnt << endl;
        os << "now totalIntervalWrCnt = " << totalIntervalWrCnt << endl;
        os << "-------------------------------------" << endl;
    }
    at::Tensor data;
    at::Tensor dataDigit;
    at::Tensor cellWrCnt;
    int64_t totalWrCnt, totalCmpWrCnt, totalIntervalWrCnt;
    int rowSize, colSize, colSize_;
    double ImaxPCell, IminPCell;
    double deltaI;
    int rightRotate, downRotate;
    torch::DeviceType device;
};


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
        static bool k = true;
        if (k)
        {
            std::cout << "get in only counters" << std::endl;
            k = false;
        }
        arrList.push_back(new phyArraySimpleEx(rowSize, colSize, toGPU));
        return arrList.size() - 1;
    }

    phyArraySimpleEx &synaccess(int x)
    {
        std::lock_guard<std::mutex> lk(mu);
        return *arrList[x];
    }

    phyArraySimpleEx &access(int x)
    {
        return *arrList[x];
    }

    phyArraySimpleEx &operator[](int x)
    {
        return *arrList[x];
    }

    void printAllInfo(std::ostream &os)
    {
        for (auto &i : arrList)
            i->print(os);
    }


    void schedule(int interval, int tim, int top_k)
    {
        static int64_t cnt = 0;
        if (cnt==0)
        {
            std::cout << "first time in schedule" << std::endl;
        }
        ++cnt;

        TORCH_INTERNAL_ASSERT(top_k*2<=arrList.size(), "top_k too large");
        if (cnt%interval == 0)
        {
            for (auto &i : arrList)
                i->rotate();

            if (cnt%tim == 0)
            {
                typedef pair<int, phyArraySimpleEx**> pv;
                vector<pv> v1, v2;
                int arr_len = arrList.size();
                for (int i=0; i<arr_len; ++i)
                    v1.push_back(make_pair(i, &arrList[i]));
                v2 = v1;
                sort(v1.begin(), v1.end(), [&](const pv &x, const pv &y) -> bool
                    {
                        return (*x.second)->totalCmpWrCnt<(*y.second)->totalCmpWrCnt;        
                    });
                
                sort(v2.begin(), v2.end(), [&](const pv &x, const pv &y) -> bool
                    {
                        return (*x.second)->totalIntervalWrCnt>(*y.second)->totalIntervalWrCnt;        
                    });

                //wr count from small to large, in v1
                //interval count from large to small, in v2

                for (int i = 0; i<top_k; ++i)
                {
                    if ((*v1[i].second)->totalCmpWrCnt<(*v2[i].second)->totalCmpWrCnt && \
                           (*v1[i].second)->totalIntervalWrCnt<(*v2[i].second)->totalIntervalWrCnt)
                    {
                        //std::cout << "swap:" << v1[i].first << "<--->" << v2[i].first << std::endl;
                        swap(arrList[v1[i].first], arrList[v2[i].first]);
                    } 
                }
                //here, we simply swap two items in arrList. This is logically right because in counter-only mode, we use realMat for computing in logic_array(pimArrayExampleCounters) and the physical array is not used for computing. So simply swapping is ok and right.

                for (int i = 0; i<top_k; ++i)
                {
                    int j = arr_len-i-1;
                    if ((*v1[j].second)->totalCmpWrCnt>(*v2[j].second)->totalCmpWrCnt && \
                           (*v1[j].second)->totalIntervalWrCnt>(*v2[j].second)->totalIntervalWrCnt)
                    {
                        //std::cout << "swap:" << v1[j].first << "<--->" << v2[j].first << std::endl;
                        swap(arrList[v1[j].first], arrList[v2[j].first]);
                    } 
                }



                for (auto &i : arrList)
                    i->clearIntervalCount();
                //std::cout << "swap over" << std::endl;
            }
        }
    }

    phyArrayManager_counters()
    {
    }
    ~phyArrayManager_counters()
    {
        for (auto &i : arrList)
        {
            delete i;
        }
    }

private:
    std::vector<phyArraySimpleEx *> arrList;
    static std::mutex mu;
};

std::mutex phyArrayManager_counters::mu;


pim_array_config decf_for_counters = {
    .rowSize = 256,
    .colSize = 256,
    .phyArrRowSize = 64,
    .phyArrColSize = 64,
    .inBits = 12,
    .outBits = 8,
    .unitBits = 16,
    .cellBits = 1,
    .has_negative_input = true,
    .max_phy_input_value = 16,
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
        torch::Tensor initMat = torch::cat({torch::zeros({phyArrRowSize, phyArrColSize+1}), torch::ones({phyArrRowSize, 1})}, 1);
        for (int i = 0; i < arrX_size; ++i)
            for (int j = 0; j < arrY_size; ++j)
            {
                int k = phyArrMan.allocPhyArray(phyArrRowSize, phyArrColSize + 2, toGPU);
                arr[i][j] = k;
                //for (int r = 0; r < phyArrRowSize; ++r)
                //{
                    //phyArrMan.synaccess(k).writeCell(r, phyArrColSize + 1, 1, torch::tensor({1}, TensorOptions(device)));
                phyArrMan.synaccess(k).initWriteMat(initMat);
                //}
            }
        ImaxPCell = phyArrMan.synaccess(0).ImaxPCell;
        IminPCell = phyArrMan.synaccess(0).IminPCell;
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
        std::cout << "------------logic array-----" << std::endl;
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
        x = x>1? 1 : x<-1? -1 : x;
        return trunc_45((x + 1) / 2 * (unitLevels - 1));
    }

    torch::Tensor unit2digit(const torch::Tensor &x)
    {
        auto larger = x>1;
        auto less = x<-1;
        auto y = x;
        y.index_put_({larger}, 1.0);
        y.index_put_({less}, -1.0);
        return y.add(1).div(2.0).mul(unitLevels - 1).add(0.5).to(torch::kInt32);
    }

    double digit2unit(int x)
    {
        return 1.0 * x / (unitLevels - 1) * 2 - 1;
    }

    torch::Tensor digit2unit(const torch::Tensor &x)
    {
        return x.to(torch::kFloat64).div(unitLevels-1).mul(2).subtract(1);
    }

    static phyArrayManager_counters phyArrMan;
private:
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
    double max_one = max_phy_input_value;


    auto index_larger = mat>max_one;
    auto index_smaller = mat<-max_one;
    auto tmp = mat;
    tmp.index_put_({index_larger}, max_one);
    tmp.index_put_({index_smaller}, -max_one);
    if (mat.size(1)<realMat.size(0))
        return torch::matmul(tmp, realMat.slice(0, 0, mat.size(1)));
    else
        return torch::matmul(tmp, realMat);
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
                            phyArrRowSize, usedcellsPerRow);
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
                            ed_arrRowId, usedcellsPerRow);
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

    if (ed_arrRowId!=0 && ed_arrColId!=0)
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
