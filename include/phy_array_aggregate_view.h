#ifndef __PHYARRAY__AGGREAGTE__
#define __PHYARRAY__AGGREAGTE__
#pragma once

#include <torch/torch.h>
#include <torch/custom_class.h>
#include "logic_array_interface.h"
#include "pim_func.h"
#include "pim_array_config.h"
#include <iostream>
#include <mutex>
#include "phy_array_simple.h"
#include <vector>
#include <memory>

using std::endl;
using std::cout;
using namespace torch::indexing;
using namespace PIM;

namespace PIM
{
/**
 * This struct is a logic aggregate view of multiple phy array.
 * if you have m*n phyarray of size [row, col], then the agg view is size [m*row, n*col]
 * You can operate m*n phyarray like one array. 
 *
 * !!!!!!! Note that the mm operation generate the results of each phyArray rather than a whole result, so that
 * then you can do ADC to the every result and add them togather for accurate simluating!!!!!!!
 * */
struct phyArrayAggrView
{
    phyArrayAggrView(int m, int n, int phyRowSize, int phyColSize, at::TensorOptions op = {}, const pim_array_pro_config *conf = &pro_decf());

    void writeMat(const std::vector<at::indexing::TensorIndex> &indices, const at::Tensor &wr_mat);
    at::Tensor readMat(const std::vector<at::indexing::TensorIndex> &indices);

    //when mm, you should convert phyAggrViewData (integer) to conductance (double).
    at::Tensor mm_2d(const at::Tensor &input_mat) {return mm_3d(input_mat.unsqueeze(0)).squeeze_(0);}     
    at::Tensor mm_3d(const at::Tensor &input_mat);  //when mm, you should convert phyAggrViewData (integer) to conductance (double).
    static at::Tensor postWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one);

    at::Tensor phyAggrViewData;  // size [m*row, n*col], type integer (means the conductance level of the cell). 
    std::vector<int> phyArrId; //store each phy array's id number.
    const pim_array_pro_config *conf;
    double deltaConduct;
    int m, n, phyArrRowSize, phyArrColSize;
};

phyArrayAggrView::phyArrayAggrView(int m, int n, int phyRowSize, int phyColSize, at::TensorOptions op, const pim_array_pro_config *conf): conf(conf), m(m), n(n), phyArrRowSize(phyRowSize), phyArrColSize(phyColSize)
{
    deltaConduct = (conf->maxConduct - conf->minConduct) / (conf->cellLevels - 1);
    if (conf->cellBits <= 8)
        phyAggrViewData = torch::zeros({m*phyRowSize, n*phyColSize}, op.dtype(torch::kUInt8));
    else if (conf->cellBits < 32)
        phyAggrViewData = torch::zeros({m*phyRowSize, n*phyColSize}, op.dtype(torch::kI32));
    else
    {
        cout << "too big cellBits, in phyArrayPro" << endl;
        throw "too big cellBits, in phyArrayPro";
    }
}


void phyArrayAggrView::writeMat(const std::vector<at::indexing::TensorIndex> &indices, const at::Tensor &wr_mat)
{
    phyAggrViewData.index_put_(indices, wr_mat);
}


at::Tensor phyArrayAggrView::readMat(const std::vector<at::indexing::TensorIndex> &indices)
{
    return phyAggrViewData.index(indices);
}

/**
 * @param input_mat: 3d tensor [batch_size, inpluses, m*phyrow], data ranges from [0, 1]
 * @return tensor: 3d tensor [batch_size, m, inpluses, n*phycol], data ranges from [0, 1]
 * */
at::Tensor phyArrayAggrView::mm_3d(const at::Tensor &input_mat)
{
    //size [m*phyrow, n*phycol] -> [m, phyrow, n*phycol]
    //auto tmp = phyAggrViewData.view({m, phyArrRowSize, -1}).to(torch::kF64);

    //input_mat's size [batch, inpluses, m*phyrow] -> size[batch, in, m, phyrow]->size [batch, m, inplses, phyrow]
    return torch::matmul(input_mat.view({input_mat.size(0), -1, m, phyArrRowSize}).permute({0, 2, 1, 3}), \
    phyAggrViewData.view({m, phyArrRowSize, -1}).to(torch::kF64).mul_(deltaConduct).add_(conf->minConduct)).div_(phyArrRowSize*conf->maxConduct);
}

/**
* @param mat: sizes {batch_size, arrX_size(m), inBits/inVbits, n*phyColSize}
* @param max_one: data will be used in post work, max_one is send from prework
* @return tensor sizes {batch_size, arrx_size(m), unitNumPerPhyRow*n} n is arry_size
*/
at::Tensor phyArrayAggrView::postWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one)
{
    double scalar_value = max_one *conf->phyArrRowSize /(conf->unitLevels-1) * conf->max_weight_value /(conf->outLevels -1) *(conf->inVLevels-1) / (conf->inLevels-1) * (conf->cellLevels-1);
    at::Tensor out;

    //now all phycol are used due to usedcellsperphyrow == phycol
    out = mat.mul(conf->outLevels - 1).round_();//.index({Ellipsis, Slice(0, conf->usedCellsPerPhyRow)}); //let out range from 0 -- outLevels -1, double
    out = out.view({mat.size(0), mat.size(1), mat.size(2), -1, conf->phyArrColSize});
    out *= conf->unitScalar.view({conf->inPluses, -1, conf->phyArrColSize}).to(mat.device());

    //output is size [batch, sizeX_size, inpluse, n*unitsPerPhyRow]
    auto output = out.view({mat.sizes()}).transpose_(2, 3).view({mat.size(0), mat.size(1), -1, conf->cellsPerUnit, conf->inPluses}).sum(3).transpose_(2, 3);

    out = (output * conf->inScalar.to(mat.device())).sum(2).mul_(scalar_value); //{batch_size, unitNumPerPhyRow}
    return out;
}

struct simpleIndex
{
    inline std::vector<at::indexing::TensorIndex> 
    operator()(int row, int ed_row, int col, int ed_col, const pim_array_pro_config *conf = &pro_decf())
    {
        return std::vector<at::indexing::TensorIndex>({Slice(row, ed_row), Slice(col*conf->cellsPerUnit, ed_col*conf->cellsPerUnit)});
    }

  /*  inline void getColPos(int col, int &Y, int &arrCol)
    {
        Y = col / conf->unitsPerPhyRow;
        arrCol = col % conf->unitsPerPhyRow * conf->cellsPerUnit;
    };*/
};
/**
 * Single instance mode for phy array manager.
 * 
 * */
struct phyArrManager
{
    static phyArrManager *getInstance()
    {
        static phyArrManager man;
        return &man;
    }

    std::shared_ptr<phyArrayAggrView> 
    allocPhyArray(int m, int n, int rowSize, int colSize, at::TensorOptions op = {}, const pim_array_pro_config *conf = &pro_decf())
    {
        std::lock_guard<std::mutex> lk(mu);
        return std::make_shared<phyArrayAggrView>(m, n, rowSize, colSize, op, conf);
    }

    phyArrManager(const phyArrManager&) = delete;
    phyArrManager& operator=(const phyArrManager&) = delete;
private:
    phyArrManager() {}
    std::mutex mu;
};

class pimArrayFast: public LogicArrayInterface
{
public:
    pimArrayFast(int rowSizeIn, int colSizeIn, const at::TensorOptions &op = {}, const pim_array_pro_config *cf = &pro_decf())
      : LogicArrayInterface(rowSizeIn, colSizeIn)
    {
        conf = cf;
        init(cf, op);
    }

    void init(const pim_array_pro_config *cf, const at::TensorOptions &op);

    void write_cell(int64_t row, int64_t col, const torch::Scalar &value) override
    {
        write_mat(torch::tensor({{value.to<double>()}}), row, col);
    }

    torch::Scalar read_cell(int64_t row, int64_t col) override
    {
        return read_mat(row, col, 1, 1).sum().item<double>();
    }

    void write_row(int64_t row, int64_t col, const at::Tensor &vec) override
    {
        write_mat(vec.view({1, -1}), row, col);
    }

    at::Tensor read_row(int64_t row, int64_t col, int64_t size) override //从x行从y列开始读取size个列的数据
    {
        return read_mat(row, col, 1, size).clone().reshape({size});
    }

    at::Tensor mv(const at::Tensor &vec) override
    {
        return mm(vec.view({-1, 1}));
    }

    at::Tensor dot_column(const at::Tensor &vec, int64_t col) override
    {
        std::cout << "currently, we do not support dot_column." << std::endl;
        throw("currently, we do not support dot_column.");
    }

    at::Tensor nmv(int64_t row, int64_t col, const torch::Scalar &value, int64_t size) override 
    {
        std::cout << "currently, we do not support nmv." << std::endl;
        throw("currently, we do not support nmv.");
    }

    void write_mat(const at::Tensor &mat) override
    {
        write_mat(mat, 0, 0);
    }

    at::Tensor read_mat() override
    {
        return read_mat(0, 0);
    }

    static const int index_len_max = std::numeric_limits<int>::max() / 2;
    at::Tensor read_mat(int row, int col, int m = index_len_max, int n = index_len_max);
    void write_mat(const at::Tensor &mat, int row, int col);

    // only mm is used in conv and linear, so we implement it first
    at::Tensor mm(const at::Tensor &mat) override;

    torch::IntArrayRef sizes() const override
    {
        return torch::IntArrayRef(sizes_vec);
    }

    at::Tensor &resize(torch::IntArrayRef size) const override
    {
        std::cout << "currently, we do not support resize." << std::endl;
        throw("currently, we do not support resize.");
    }

    std::ostream &print(std::ostream &os) const override
    {
        return os;
    }

protected:
    const pim_array_pro_config *conf;
    int arrX_size, arrY_size, phyAllRowSize;
    std::vector<int64_t> sizes_vec;
    std::shared_ptr<phyArrayAggrView> arr, narr;
    at::TensorOptions op;
};

void pimArrayFast::init(const pim_array_pro_config *cf, const torch::TensorOptions &op)
{
    if (cf->unitBits % cf->cellBits != 0)
    {
        throw "unitBits mod cellBits != 0";
    }

    if (cf->inBits % cf->inVBits != 0)
    {
        throw "inBitsm mod inVBits!=0";
    }
    if (cf->usedCellsPerPhyRow!=cf->phyArrColSize)
    {
        cout << "we now don't cf->usedCellsPerPhyRow!=cf->phyArrColSize";
        throw "cf->usedCellsPerPhyRow!=cf->phyArrColSize";
    }

    // one phy array row can store how many units
    if (cf->unitsPerPhyRow <= 0)
    {
        throw "unitsPerPhyRow<=0";
    }

    arrX_size = trunc_ceil((int)rowSize, cf->phyArrRowSize);
    arrY_size = trunc_ceil((int)colSize, cf->unitsPerPhyRow);

    phyAllRowSize = arrX_size*cf->phyArrRowSize;
    this->op = op;

    arr = phyArrManager::getInstance()->allocPhyArray(arrX_size, arrY_size, cf->phyArrRowSize, cf->phyArrColSize, op, cf);
    narr = phyArrManager::getInstance()->allocPhyArray(arrX_size, arrY_size, cf->phyArrRowSize, cf->phyArrColSize, op, cf);

    if (cf->mode == 1) // last col is 0, ref colum is set different from pimArrayExample, be careful
    {
        cout << "don't support ref column mode now" << endl;
        throw "don't support ref column mode";
    }
    sizes_vec = std::vector<int64_t>({rowSize, colSize});
}

at::Tensor pimArrayFast::read_mat(int row, int col, int m, int n)
{
    int ed_row = std::min((int64_t)row + m, rowSize);
    int ed_col = std::min((int64_t)col + n, colSize);

    m = ed_row - row;
    n = ed_col - col;
    auto pos_cell = arr->readMat(simpleIndex()(row, ed_row, col, ed_col, conf));
    auto neg_cell = narr->readMat(simpleIndex()(row, ed_row, col, ed_col, conf));

    auto cell2digit = [&](at::Tensor &in) -> at::Tensor
    {
        at::Tensor data = torch::zeros({m, n}, op.dtype(torch::kI32));

        for (int i = 0; i < conf->cellsPerUnit; ++i)
        {
            data.add_(in.index({Slice(), Slice(i, n * conf->cellsPerUnit, conf->cellsPerUnit)}).__lshift__(i * conf->cellBits));
        }

        return data;
    };

    auto digit2unit = [&](const at::Tensor &in) -> at::Tensor
    {
        return in.to(torch::kFloat64).mul(conf->max_weight_value / (conf->unitLevels - 1));
    };

    return digit2unit(cell2digit(pos_cell)) - digit2unit(cell2digit(neg_cell));
}

/**
 * @param mat: 2d tensor
 * @param row: start row
 * @param col: start col
 */
void pimArrayFast::write_mat(const at::Tensor &matin, int row, int col)
{
    at::Tensor mat = matin.detach();
    int64_t m = mat.size(0), n = mat.size(1);
    int ed_row = std::min(row + m, rowSize);
    int ed_col = std::min(col + n, colSize);

    //if (conf->mode == 0) // positive array & negative array
    auto unit2digit = [&](const at::Tensor &x, at::Tensor &pos, at::Tensor &neg) -> void
    {
        pos = x.div(conf->max_weight_value);
        pos.index_put_({pos > 1}, 1.0);
        pos.index_put_({pos < -1}, -1.0); // -1~~1

        pos = pos.mul(conf->unitLevels - 1); // -(unitLevels -1 ) ~~~~ (unitLevels -1 )
        neg = pos.clone();

        pos.index_put_({pos < 0}, 0); // 0 ~ unitLevels-1
        neg.index_put_({neg > 0}, 0);
        neg.abs_(); // 0 ~ unitLevels-1
        pos = pos.round_().to(torch::kInt32);
        neg = neg.round_().to(torch::kInt32);
    };

    int mask = (conf->cellLevels) - 1;
    auto digit2cell = [&](at::Tensor &in, at::Tensor &out) -> void
    {
        out = torch::cat({std::vector<at::Tensor>(conf->cellsPerUnit, in.view({m, n, 1}))}, 2);
        out.__irshift__(conf->crshift.to(in.device())).bitwise_and_(mask);
        out = out.view({m, n * conf->cellsPerUnit});
    };
    at::Tensor neg, pos, pos_cell, neg_cell;

    unit2digit(mat, pos, neg); //pos {}
    digit2cell(pos, pos_cell);
    digit2cell(neg, neg_cell);

    arr->writeMat(simpleIndex()(row, ed_row, col, ed_col, conf), pos_cell);
    narr->writeMat(simpleIndex()(row, ed_row, col, ed_col, conf), neg_cell);
}

/**
 * Performs mm.
* @param mat input matrix, should be 2D or 3D Tensor of sizes {batch_size, rowSize} or {batch_size, row_size, col_size}
* @return 2d tensor of sizes {batch_size, colSize}
*/
at::Tensor pimArrayFast::mm(const at::Tensor &matin)
{
    at::Tensor mat = matin.detach();

    auto mm2d = [&](const at::Tensor &mat) -> at::Tensor {
        int batch_size = mat.size(0);
        int siz = mat.size(1);
        double max_one;
        auto mat_out = mat;
        if (siz<phyAllRowSize)
        {
            mat_out = torch::cat({mat, torch::zeros({batch_size, phyAllRowSize-siz}, op.dtype(torch::kF64))}, 1);
        }
        at::Tensor input = phyArrayPro::preWorkForMM(mat_out, conf, std::ref(max_one)); // input should be tensor of size {batch_size, inBits/inVBits, rowSize}

        at::Tensor out;// = torch::zeros({arrX_size, batch_size, arrY_size * conf->unitsPerPhyRow}, op.dtype(torch::kF64));

        // postive & negative array mode
        if (conf->mode == 0)
        {
            at::Tensor nout;// = torch::zeros({arrX_size, batch_size, arrY_size * conf->unitsPerPhyRow}, op.dtype(torch::kF64));
            out = phyArrayAggrView::postWorkForMM(arr->mm_3d(input), conf, max_one);
            nout = phyArrayAggrView::postWorkForMM(narr->mm_3d(input), conf, max_one);//[batch, arrX_size, inpluses, arrY_size*phyCol]->[batch, arrX, arrY*unitsPerPhyRow]

            out.subtract_(nout);
        }
        else // ref col mode
        {
        }
        return out.sum(1).index({Ellipsis, Slice(0, colSize)});
    };

    if (mat.sizes().size() == 2)
    {
        return mm2d(mat);
    }
    else
    {            
        int batch_size, row_size, col_size;
        batch_size = mat.size(0);
        row_size = mat.size(1);
        col_size = mat.size(2);
        return mm2d(mat.reshape({-1, col_size})).reshape({batch_size, row_size, -1});
    }
}


}
#endif

