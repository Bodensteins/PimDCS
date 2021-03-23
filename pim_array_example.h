#ifndef PIMTORCH_ARRAY_EXAMPLE_H
#define PIMTORCH_ARRAY_EXAMPLE_H

#pragma once

#include "pim_array_config.h"
#include "phy_array_simple.h"
#include "logic_array_interface.h"
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

    phyArraySimple &synaccess(int x)
    {
        std::lock_guard<std::mutex> lk(mu);
        return *arrList[x];
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

std::mutex phyArrayManager::mu;


class pimArrayExample : public LogicArrayInterface
{
public:
   pimArrayExample(int rowSizeIn, int colSizeIn, const torch::TensorOptions& op = {}, pim_array_config cf = decf): LogicArrayInterface(rowSizeIn, colSizeIn)
   {
       cf.rowSize = rowSizeIn;
       cf.colSize = colSizeIn;
       init(cf, op);
   }

    pimArrayExample(const pim_array_config &cf, const torch::TensorOptions &op = {}): LogicArrayInterface(cf.rowSize, cf.colSize)
    {
        init(cf, op);
    }

    void init(const pim_array_config &cf, const torch::TensorOptions &op)
    {
        phyArrRowSize = cf.phyArrRowSize;
        phyArrColSize = cf.phyArrColSize;
        assert(cf.cellBits == 1);

        // one phy array row can store how many units
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
        torch::Tensor initMat = torch::cat({torch::zeros({phyArrRowSize, phyArrColSize+1}, torch::kInt8), torch::ones({phyArrRowSize, 1}, torch::kInt8)}, 1).to(op.device());
        for (int i = 0; i < arrX_size; ++i)
            for (int j = 0; j < arrY_size; ++j)
            {
                int k = phyArrMan.allocPhyArray(phyArrRowSize, phyArrColSize + 2, toGPU);
                arr[i][j] = k;
                //for (int r = 0; r < phyArrRowSize; ++r)
                  //  phyArrMan.synaccess(k).writeCell(r, phyArrColSize + 1, 1, torch::tensor({1}, TensorOptions(device)));
                phyArrMan.synaccess(k).initWriteMat(initMat);
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
        return os;
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

phyArrayManager pimArrayExample::phyArrMan;

/*
 * write_cell to [row, col] 
 *
 * */
void pimArrayExample::write_cell(int64_t row, int64_t col, const torch::Scalar &value)
{
    int arrX, arrY, arrRowId, arrColId;

    getPosition(row, col, arrX, arrY, arrRowId, arrColId);
    int d = unit2digit(value.to<double>());
    at::Tensor data = torch::empty({unitBits}, TensorOptions(device).dtype(torch::kInt8));
    for (int i = 0; i < unitBits; ++i)
        data[i] = (d >> i) & 1;
    phyArrMan[arr[arrX][arrY]].writeCell(arrRowId, arrColId, unitBits, data);
}

/*
 *  read_cell from [row, col]
 * */
torch::Scalar pimArrayExample::read_cell(int64_t row, int64_t col)
{
    int arrX, arrY, arrRowId, arrColId;

    getPosition(row, col, arrX, arrY, arrRowId, arrColId);

    at::Tensor data;
    int x = 0;
    phyArrMan[arr[arrX][arrY]].readCell(arrRowId, arrColId, unitBits, data);
    for (int i = 0; i < unitBits; ++i)
        x = (data[i].item<int>() << i) | x;
    return digit2unit(x);
}

void pimArrayExample::write_row(int64_t row, int64_t col, const torch::Tensor &vec)
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

torch::Tensor pimArrayExample::read_row(int64_t row, int64_t col, int64_t size) //从x行从y列开始读取size个列的数据
{
    int col_ed = col + size;

    int arrX, st_arrY, ed_arrY, arrRowId, st_arrColId, ed_arrColId;
    at::Tensor data;

    getRowPos(row, arrX, arrRowId);
    getColPos(col, st_arrY, st_arrColId);
    getColPos(col_ed, ed_arrY, ed_arrColId);

    if (col == col_ed)
    {
        phyArrMan[arr[arrX][st_arrY]].readCell(arrRowId, st_arrColId, ed_arrColId - st_arrColId, data);   
    }
    else
    {
        int ed = usedcellsPerRow - st_arrColId;
        at::Tensor out;
        data = torch::empty({size*unitBits}, TensorOptions(device).dtype(torch::kInt8));

        phyArrMan[arr[arrX][st_arrY]].readCell(arrRowId, st_arrColId, ed, out);
        data.index_put_({torch::indexing::Slice(0, ed)}, out);

        for (int y = st_arrY + 1; y < ed_arrY; ++y)
        {
            phyArrMan[arr[arrX][y]].readCell(arrRowId, 0, usedcellsPerRow, out);
            data.index_put_({torch::indexing::Slice(ed, ed + usedcellsPerRow)}, out);
            ed += usedcellsPerRow;
        }
        if (ed_arrColId != 0)
        {
            phyArrMan[arr[arrX][ed_arrY]].readCell(arrRowId, 0, ed_arrColId, out);

            data.index_put_({torch::indexing::Slice(ed, ed + ed_arrColId)}, out);
        }
    }
    at::Tensor out=torch::zeros({size}, TensorOptions(device).dtype(torch::kFloat64));
    for (int i=0; i<size; ++i)
    {
        int d = 0;
        for (int j=0; j<unitBits; ++j)
            d = (data[i*unitBits+j].item<int>() <<j)|d;
        out[i] = digit2unit(d);
    }

    return out;
}

/*
 *  input to digit
 *  when perform VMM operation, we need to revert input value to digit pulse.
 *
 *  if input vec is 1 dim, e.g. input vector [a1 a2] with inbits = 2, output a matrix of  | d1#0 d1#1 |,     di means the digit value of ai
 *                                                                                        | d2#0 d2#1 |      di#j, means jth bits of di.
 *  if input vec if 2 dim, e.g. |a1 b1|, this is a batch with size 2. batch 1 is [a1, a2], batch 2 is [b1, b2]
 *                              |a2 b2|
 *  output a tensor with 3 dim, its first dim is batch size. and other two dim is similar to input vec with 1 dim.   
 *
 * */
at::Tensor pimArrayExample::input2digit(const at::Tensor &vec, double &max_one)
{
    if (dynamic_max_input)
    {
        if (has_negative_input) 
            max_one = std::min(max_phy_input_value, vec.abs().max().item<double>());
        else
            max_one = std::min(max_phy_input_value, vec.max().item<double>()); 
        // std::cout << "max_one = " << max_one << std::endl;
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

/*
 *  matric * vector
 *
 *
 * */
torch::Tensor pimArrayExample::mv(const torch::Tensor &vec)
{
    double max_one;
    at::Tensor vecDigit;
    const static double scalar_value = maxIsumPerPhyCol/(ImaxPCell-IminPCell)/(unitLevels-1)/(outLevels-1)/(inLevels-1);
    const static double neg = has_negative_input? -1 : 1;

    at::Tensor v = vec;
    if (v.size(0)<phyAllRowSize)
    {
        v.resize_({phyAllRowSize});
        v.index_put_({Slice(rowSize, phyAllRowSize)}, 0);
    }
    vecDigit = input2digit(v, max_one).to(torch::kFloat64);

    /*
     *  A logic arrays is formed by multiple phy arrays, its row has arrX_size phyArrays and its col has arrY_size phyArrays.
     *  there migit be unsed col in each phyArrays, so Iref's 3th value is used cells per row. 
     *  As this class implement input value to digit pulse, so inBits will revert to #inBits pulses. Every pulse need to do one mv operation in phy arrays.
     *  So its 4th value in Iout is inBits. 
     * */
    at::Tensor Iout = torch::zeros({arrX_size, arrY_size, usedcellsPerRow, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    at::Tensor out_digit = torch::zeros({arrX_size, arrY_size, unitNumPerRow, inBits}, TensorOptions(device).dtype(torch::kInt64));
    at::Tensor Iref0 = torch::zeros({arrX_size, arrY_size, 1, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    at::Tensor Iref1 = torch::zeros({arrX_size, arrY_size, 1, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    auto outADC=[&](const at::Tensor &x)->at::Tensor
    {   
        return x.mul((outLevels-1)/maxIsumPerPhyCol).add(0.5).to(torch::kInt64);
    };
    
    /*
     *  record current out of each phy array by Iout
     *  Ire0 and Iref1 means ref column, please read our manual for more information about ref column.
     *
     * */
    for (int i=0; i<arrX_size; ++i)
    {
        for (int j=0; j<arrY_size; ++j)
        {
            at::Tensor tmp_out;
            phyArrMan[arr[i][j]].vmm(vecDigit.slice(0, i*phyArrRowSize, (1+i)*phyArrRowSize), tmp_out);
            Iout[i][j] = tmp_out.slice(0, 0, usedcellsPerRow);
            Iref0[i][j][0] = tmp_out[phyArrColSize];
            Iref1[i][j][0] = tmp_out[phyArrColSize+1];
        }
    }

    /*
     *  use ref column to Numerate final output
     * */
    at::Tensor Iref0_digit = outADC(Iref0);
    at::Tensor refValue = outADC(Iref1)-Iref0_digit;
    refValue = refValue.__lshift__(unitBits).subtract(refValue);

    Iout = outADC(Iout) - Iref0_digit;

    /*
     *  use unit scalar to revert digit output to double type
     *  here, we set up scalar for one unit's different bits.
     * */
    at::Tensor unitScalar = torch::empty({usedcellsPerRow, inBits}, TensorOptions(device).dtype(torch::kInt64));
    for (int i=0; i<unitBits; ++i)
    {
        unitScalar.index_put_({Slice(i, usedcellsPerRow, unitBits)}, 1 <<i);
        
    }

    Iout = Iout * unitScalar;


    for (int i=0; i<unitNumPerRow; ++i)
    {
        out_digit.index_put_({Slice(), Slice(), Slice(i, i+1)}, Iout.index({Slice(), Slice(), Slice(i*unitBits, (i+1)*unitBits)}).sum(2, true));
    }
    

    out_digit = out_digit.__lshift__(1)-refValue;

    unitScalar = torch::empty({inBits}, TensorOptions(device).dtype(torch::kFloat64));

    for (int i=0; i<inBits-1; ++i)
    {
        unitScalar.index_put_({Slice(i, i+1)}, (1 <<i)*scalar_value);
    }

    unitScalar.index_put_({Slice(inBits-1)}, (1 <<(inBits-1))*scalar_value*neg);
    at::Tensor out = (out_digit.to(torch::kFloat64)*unitScalar*max_one).sum(3).sum(0).view(-1);

    return out.index({Slice(0, colSize)});
}

torch::Tensor pimArrayExample::dot_column(const torch::Tensor &vec, int64_t col)
{
    assert("not implement dot_colume"!="not implement dot_colume");
}

torch::Tensor pimArrayExample::nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) //数值乘以向量，需要输入操作的row号
{   
    //assert("not_implement");
    assert(false);
}

/*
 *  similar to MV function. but input param is a mat with dim 1 of batch_size.
 *
 * */
torch::Tensor pimArrayExample::mm(const torch::Tensor &mat) 
{
    double max_one;
    int len = mat.size(0);
    at::Tensor vecDigit;
    const double scalar_value = maxIsumPerPhyCol/(ImaxPCell-IminPCell)/(unitLevels-1)/(outLevels-1)/(inLevels-1);
    const double neg = has_negative_input? -1 : 1;

    at::Tensor v = mat.t();
    // std::cout << v << std::endl;
    // std::cout << "in1" << std::endl;
    if (v.size(0)<phyAllRowSize)
    {
        auto tmp = torch::zeros({phyAllRowSize-v.size(0), v.size(1)}, TensorOptions(device).dtype(torch::kFloat64));
        // std::cout << tmp << std::endl;
        // std::cout << (device == torch::kCUDA) << std::endl;
        v = torch::cat({v, tmp});
        v.index_put_({Slice(v.size(0), phyAllRowSize)}, 0);
    }
    // std::cout << v << std::endl;
    // std::cout << "in2" << std::endl;
    vecDigit = input2digit(v, max_one).to(torch::kFloat64);

    at::Tensor Iout = torch::zeros({arrX_size, arrY_size, len, usedcellsPerRow, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    at::Tensor out_digit = torch::zeros({arrX_size, arrY_size, len, unitNumPerRow, inBits}, TensorOptions(device).dtype(torch::kInt64));
    at::Tensor Iref0 = torch::zeros({arrX_size, arrY_size, len, 1, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    at::Tensor Iref1 = torch::zeros({arrX_size, arrY_size, len, 1, inBits}, TensorOptions(device).dtype(torch::kFloat64));
    
    auto outADC=[&](const at::Tensor &x)->at::Tensor
    {   
        return x.mul((outLevels-1)/maxIsumPerPhyCol).add(0.5).to(torch::kInt64);
    };

    for (int i=0; i<arrX_size; ++i)
    {
        at::parallel_for(0, arrY_size, 0, [&](int start, int end) {
            for (int j=start; j<end; ++j)
            {
                at::Tensor tmp_out;
                phyArrMan[arr[i][j]].vmm(vecDigit.index({Slice(), Slice(i*phyArrRowSize, (1+i)*phyArrRowSize)}), tmp_out);
                
                Iout[i][j] = tmp_out.index({Slice(), Slice(0, usedcellsPerRow)});
                Iref0[i][j] = tmp_out.index({Slice(), Slice(phyArrColSize, phyArrColSize+1)});
                
                Iref1[i][j]= tmp_out.index({Slice(), Slice(phyArrColSize+1, phyArrColSize+2)});
            }
        });
    }
    // std::cout << "in3" << std::endl;
    at::Tensor Iref0_digit = outADC(Iref0);
    at::Tensor refValue = outADC(Iref1)-Iref0_digit;
    refValue = refValue.__lshift__(unitBits).subtract(refValue);

    Iout = outADC(Iout) - Iref0_digit;
    //std::cout << Iout << std::endl;
    at::Tensor unitScalar = torch::empty({usedcellsPerRow, inBits}, TensorOptions(device).dtype(torch::kInt64));
    //std::cout << "---" << std::endl;
    for (int i=0; i<unitBits; ++i)
    {
        unitScalar.index_put_({Slice(i, usedcellsPerRow, unitBits)}, 1 <<i);
    }
    
    Iout = Iout * unitScalar;
    // std::cout << "in4" << std::endl;
    //std::cout << Iout << std::endl;
    for (int i=0; i<unitNumPerRow; ++i)
    {
        out_digit.index_put_({Slice(), Slice(), Slice(), Slice(i, i+1)}, Iout.index({Slice(), Slice(), Slice(), Slice(i*unitBits, (i+1)*unitBits)}).sum(3, true));
    }
    
    
    out_digit = out_digit.__lshift__(1)-refValue;
    //std::cout << out_digit << std::endl;
    unitScalar = torch::empty({inBits}, TensorOptions(device).dtype(torch::kFloat64));

    for (int i=0; i<inBits-1; ++i)
    {
        unitScalar.index_put_({Slice(i, i+1)}, (1 <<i)*scalar_value);
    }
    
    // std::cout << "in5" << std::endl;
    unitScalar.index_put_({Slice(inBits-1)}, (1 <<(inBits-1))*scalar_value*neg);
    
    at::Tensor out = (out_digit.to(torch::kFloat64)*unitScalar*max_one).sum(4).sum(0);
    

    at::Tensor real_out = torch::empty({len, colSize}, TensorOptions(device).dtype(torch::kFloat64));
    for (int i=0; i<len; ++i)
        real_out[i] = out.index({Slice(), Slice(i, i+1)}).reshape({-1}).slice(0, 0, colSize);
    return real_out;
}


/*
*   in face, now we only use write_mat in pim_linear & pim_conv.
*   so we optimizer it first. and then come to write_row & write_cell
*/
void pimArrayExample::write_mat(const torch::Tensor &mat) 
{
    int M = std::min(mat.size(0), rowSize);
    int N = std::min(mat.size(1), colSize);
    // at::parallel_for(0, M, 0, [&](int st, int ed)->void
    // {
    //     for (int i=st; i<ed; ++i)
    //         write_row(i, 0, mat[i]);
    // });

    torch::Tensor data = unit2digit(mat).index({Slice(0, M), Slice(0, N)});
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
    

torch::Tensor pimArrayExample::read_mat() 
{
    at::Tensor out=torch::empty({rowSize, colSize}, TensorOptions(device).dtype(torch::kFloat64));
    at::parallel_for(0, rowSize, 0, [&](int st, int ed)->void
    {
        for (int i=st; i<ed; ++i)
            out[i] = read_row(i, 0, colSize); 
    });
    return out;
}
#endif
