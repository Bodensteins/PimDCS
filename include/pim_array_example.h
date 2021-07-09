#ifndef PIMTORCH_ARRAY_EXAMPLE_H
#define PIMTORCH_ARRAY_EXAMPLE_H

#pragma once

#include "pim_array_config.h"
#include "phy_array_simple.h"
#include "logic_array_interface.h"
#include "pim_func.h"
#include <cassert>
#include <mutex>
#include <algorithm>
#include <map>
#include <vector>

using PIM::LogicArrayInterface;
using PIM::SimpleLogicArray;
using torch::TensorOptions;
using torch::indexing::Ellipsis;
using torch::indexing::None;
using torch::indexing::Slice;
using namespace PIM;


class phyArrayManagerPro
{
public:
    int allocPhyArray(int rowSize, int colSize, at::TensorOptions op = {}, const pim_array_pro_config *conf = &pro_decf())
    {
        std::lock_guard<std::mutex> lk(mu);
        arrList.push_back(new phyArrayPro(rowSize, colSize, op, conf));
        return arrList.size() - 1;
    }

    std::pair<int, int> allocPhyArray(int n, int rowSize, int colSize, at::TensorOptions op = {}, const pim_array_pro_config *conf = &pro_decf())
    {
        std::lock_guard<std::mutex> lk(mu);
        for (int i=0; i<n; ++i)
            arrList.push_back(new phyArrayPro(rowSize, colSize, op, conf));
        return std::make_pair((int)arrList.size() - n, (int)arrList.size() - 1);
    }

    phyArrayPro &synaccess(int x)
    {
        std::lock_guard<std::mutex> lk(mu);
        return *arrList[x];
    }

    phyArrayPro &access(int x)
    {
        return *arrList[x];
    }

    phyArrayPro &operator[](int x)
    {
        return *arrList[x];
    }

    void printAllInfo(std::ostream &os)
    {
        for (auto &i : arrList)
            i->print(os);
    }

    void printArea() {
      double total_phy_arrays_area = (pro_decf().cell_area * 1e-6 * pro_decf().phyArrRowSize *
          pro_decf().phyArrColSize + pro_decf().array_peripheral) * arrList.size();
      double total_row_inf_area = (ceil(arrList.size() / pro_decf().row_share_subarray)) * pro_decf().DAC_area * 1e-6;
      double total_col_inf_area = (ceil(arrList.size() / pro_decf().col_share_subarray)) * pro_decf().ADC_area * 1e-6;
      double total_area = total_phy_arrays_area + total_row_inf_area + total_col_inf_area + (pro_decf().share_peripheral * 1e-6);
      std::cout << "Area info:\n"
                << "Physical array area: " << total_phy_arrays_area << " mm^2\n"
                << "DAC area: " << total_row_inf_area << " mm^2\n"
                << "ADC area: " << total_col_inf_area << " mm^2\n"
                << "Total area: " << total_area << " mm^2\n";
    }

    ~phyArrayManagerPro()
    {
        for (auto &i : arrList)
        {
            delete i;
        }
    }

/**
 *  @param time_ns: latency_add in unit of nano second
 *  @param type: 0 --> mm latency added,  1 --> write latency added, 2 --> read latency added
 */
    void latency_add(double time_ns, int type)
    {
        lat[type].latency_add(time_ns);
    }

    void print_latency(std::ostream &os)
    {
        pim_latency all;
        //currently, read latency is not calculated
        for (int i=0; i<2; ++i)
            all.latency_add(lat[i]);
        os << "model write latency = ";
        lat[1].print_latency(os);
        os << "\r\nmodel mm latency = ";
        lat[0].print_latency(os);
        os << "\r\nmodel running (all) latency = ";
        all.print_latency(os);
        os << std::endl;
    }

    //type 0 -> read, write #operation
    //type 1 -> mm #operation 
    void op_add(int64_t x, int type)
    {
        op[type].count_add(x);
    }

    void print_op(std::ostream &os)
    {
        os << "memory operation count = " << op[0].count  << " (" <<  (double)op[0].count/1e9 << "G)" << std::endl;
        os << "cal operation count = " << op[1].count << " (" << (double)op[1].count/1e9 << "G)" << std::endl;

    }

    void print_power_efficiency(std::ostream &os)
    {
        os << "GOPS/W means giga operations per second per watt" << endl;
        os << "memory op power efficiency = " << (double)op[0].count/(get_read_energy()+get_write_energy()) << " GOPS/W" << std::endl;

        os << "calculation op power efficiency = " << (double)op[1].count/(get_compute_energy()) << " GOPS/W" << std::endl;


        os << "total op power efficiency = " << (double)(op[0].count+op[1].count)/(get_total_energy()) << " GOPS/W" << std::endl;
    }

    double get_total_energy()
    {
        return get_read_energy() + get_write_energy() + get_compute_energy();
    }

//    double get_array_energy()
//    {
//        return array_energy;
//    }
//
//    void add_array_energy(double deltaE)
//    {
//        std::lock_guard<std::mutex> lk(mu);
//
//        array_energy += deltaE;
//    }
//
//    double get_periphery_circuit_energy()
//    {
//        return periphery_circuit_energy;
//    }
//
//    void add_periphery_circuit_energy(double deltaE)
//    {
//        std::lock_guard<std::mutex> lk(mu);
//
//        periphery_circuit_energy += deltaE;
//    }

    double get_read_energy()
    {
        //todo:
        double readEnergy = 0;

        for (auto &i : arrList)
        {
            readEnergy += i->readEnergy;
        }

        return readEnergy;
    }

//    void add_read_energy(double deltaE)
//    {
//        std::lock_guard<std::mutex> lk(mu);
//
//        read_energy += deltaE;
//    }

    double get_write_energy()
    {
        double writeEnergy = 0;

        for (auto &i : arrList)
        {
            writeEnergy += i->writeEnergy;
        }

        return writeEnergy;
    }

//    void add_write_energy(double deltaE)
//    {
//        std::lock_guard<std::mutex> lk(mu);
//
//        write_energy += deltaE;
//    }

    double get_compute_energy()
    {
        //todo:
        double computeEnergy = adder_energy;

        for (auto &i : arrList)
        {
            computeEnergy += i->computeEnergy;
        }

        return computeEnergy;
    }

//    void add_compute_energy(double deltaE)
//    {
//        std::lock_guard<std::mutex> lk(mu);
//
//        compute_energy += deltaE;
//    }

    void add_adder_energy(double deltaE)
    {
        std::lock_guard<std::mutex> lk(mu);

        adder_energy += deltaE;
    }

private:
    std::vector<phyArrayPro *> arrList;
    static std::mutex mu;
    pim_latency lat[3]; //0-> mm_latency, 1->wr_latency, 2->rd_latency
    operation_count op[2];//0-> read/write #operation,  1-> calculation #operation
    //energy info
    double adder_energy;
//    double array_energy;
//    double periphery_circuit_energy;
//    double read_energy;
//    double write_energy;
//    double compute_energy;
};

std::mutex phyArrayManagerPro::mu;

class pimArrayPro: public LogicArrayInterface
{
public:
    pimArrayPro(int rowSizeIn, int colSizeIn, const torch::TensorOptions &op = {}, const pim_array_pro_config *cf = &pro_decf())
      : LogicArrayInterface(rowSizeIn, colSizeIn)
    {
        conf = cf;
        init(cf, op);
    }

    void init(const pim_array_pro_config *cf, const torch::TensorOptions &op);

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

    at::Tensor read_mat(int row, int col, int m = phyArrayPro::index_len_max, int n = phyArrayPro::index_len_max);
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
        if (conf->mode == 1)
        {
            for (int i = 0; i < arrX_size; ++i)
                for (int j = 0; j < arrY_size; ++j)
                    phyArrManPro[arr[i][j]].print(os);
        }
        else
        {
            for (int i = 0; i < arrX_size; ++i)
                for (int j = 0; j < arrY_size; ++j)
                    phyArrManPro[arr[i][j]].print(os);
            for (int i = 0; i < arrX_size; ++i)
                for (int j = 0; j < arrY_size; ++j)
                    phyArrManPro[narr[i][j]].print(os);
        }

        return os;
    }


    template <typename F>
    void locate(int row, int col, int ed_row, int ed_col, F fun);
    static phyArrayManagerPro phyArrManPro;
protected:
    std::vector<int64_t> sizes_vec;
    const pim_array_pro_config *conf;
    int arrX_size, arrY_size;
    int phyAllRowSize;
    std::vector<std::vector<int>> arr, narr;
    torch::TensorOptions op;
};

phyArrayManagerPro pimArrayPro::phyArrManPro;

void pimArrayPro::init(const pim_array_pro_config *cf, const torch::TensorOptions &op)
{
    if (cf->unitBits % cf->cellBits != 0)
    {
        throw "unitBits mod cellBits != 0";
    }

    if (cf->inBits % cf->inVBits != 0)
    {
        throw "inBitsm mod inVBits!=0";
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

    arr = std::vector<std::vector<int>>(arrX_size, std::vector<int>(arrY_size));

    if (cf->mode == 0)
        narr = arr;
    at::Tensor initMat;
    auto type = cf->cellBits <= 8 ? torch::kUInt8 : torch::kInt32;

    if (cf->mode == 1) // last col is 0, ref colum is set different from pimArrayExample, be careful
        initMat = torch::cat({torch::zeros({cf->phyArrRowSize, cf->phyArrColSize}, op.dtype(type)), torch::ones({cf->phyArrRowSize, 1}, op.dtype(type)),
                              torch::zeros({cf->phyArrRowSize, 1}, op.dtype(type))},
                             1);
    else
        initMat = torch::zeros({cf->phyArrRowSize, cf->phyArrColSize}, op.dtype(type));

    for (int i = 0; i < arrX_size; ++i)
        for (int j = 0; j < arrY_size; ++j)
        {
            if (cf->mode == 1) // ref  col mode
            {
                int k = phyArrManPro.allocPhyArray(cf->phyArrRowSize, cf->phyArrColSize + 2, op, conf);
                arr[i][j] = k;
                phyArrManPro.synaccess(k).writeMat(initMat);
            }
            else // postive & negative array mode
            {
                int k = phyArrManPro.allocPhyArray(cf->phyArrRowSize, cf->phyArrColSize, op, conf);
                arr[i][j] = k;
                phyArrManPro.synaccess(k).writeMat(initMat);

                narr[i][j] = k = phyArrManPro.allocPhyArray(cf->phyArrRowSize, cf->phyArrColSize, op, conf);
                phyArrManPro.synaccess(k).writeMat(initMat);
            }
        }

    sizes_vec = std::vector<int64_t>({rowSize, colSize});

    // for (int i=0; i<phyAllRowSize; ++i)
    //     write_row(i, 0, torch::zeros({colSize}, torch::kFloat64));
}

at::Tensor pimArrayPro::read_mat(int row, int col, int m, int n)
{
    int ed_row = std::min((int64_t)row + m, rowSize);
    int ed_col = std::min((int64_t)col + n, colSize);

    m = ed_row - row;
    n = ed_col - col;

    auto getColPos = [&](int col, int &Y, int &arrCol) -> void {
        Y = col / conf->unitsPerPhyRow;
        arrCol = col % conf->unitsPerPhyRow * conf->cellsPerUnit;
    };

    auto getRowPos = [&](int row, int &X, int &arrRow) -> void {
        X = row / conf->phyArrRowSize;
        arrRow = row % conf->phyArrRowSize;
    };

    if (conf->mode == 0) // positive array & negative array
    {
        int ed_arrY, ed_arrColId;
        int ed_arrX, ed_arrRowId;
        int st_arrX, st_arrRowId;
        int st_arrY, st_arrColId;
        int lenx, leny;

        at::Tensor pos_cell = torch::empty({m, conf->cellsPerUnit * n}, op.dtype(torch::kI32));

        at::Tensor neg_cell = torch::empty({m, conf->cellsPerUnit * n}, op.dtype(torch::kI32));

        getRowPos(row, st_arrX, st_arrRowId);
        getRowPos(ed_row, ed_arrX, ed_arrRowId);

        getColPos(col, st_arrY, st_arrColId);
        getColPos(ed_col, ed_arrY, ed_arrColId);

        leny = ed_arrY - st_arrY + 1;
        lenx = ed_arrX - st_arrX + 1;
        at::parallel_for(0, leny * lenx, 0, [&](int st, int ed) -> void {
            for (int k = st; k < ed; ++k)
            {
                int i = k / leny;
                int j = k % leny;
                int X, Y, x, y, m, n;

                if (i == 0)
                    X = 0, x = st_arrRowId, m = lenx == 1 ? ed_arrRowId - st_arrRowId : conf->phyArrRowSize - st_arrRowId;
                else if (i == lenx - 1)
                    X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = ed_arrRowId;
                else
                    X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = conf->phyArrRowSize;

                if (j == 0)
                    Y = 0, y = st_arrColId, n = leny == 1 ? ed_arrColId - st_arrColId : conf->usedCellsPerPhyRow - st_arrColId;
                else if (j == leny - 1)
                    Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = ed_arrColId;
                else
                    Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = conf->usedCellsPerPhyRow;
                if (m == 0 || n == 0)
                    continue;
                pos_cell.index({Slice(X, X + m), Slice(Y, Y + n)}) =
                    phyArrManPro[arr[st_arrX + i][st_arrY + j]].readMat(x, y, m, n);
                neg_cell.index({Slice(X, X + m), Slice(Y, Y + n)}) =
                    phyArrManPro[narr[st_arrX + i][st_arrY + j]].readMat(x, y, m, n);
            }
        });
        auto cell2digit = [&](at::Tensor &in) -> at::Tensor {
            at::Tensor data = torch::zeros({m, n}, op.dtype(torch::kI32));

            for (int i = 0; i < conf->cellsPerUnit; ++i)
            {
                data.add_(in.index({Slice(), Slice(i, n * conf->cellsPerUnit, conf->cellsPerUnit)}).__lshift__(i * conf->cellBits));
            }

            return data;
        };

        auto digit2unit = [&](const at::Tensor &in) -> at::Tensor {
            return in.to(torch::kFloat64).mul(conf->max_weight_value / (conf->unitLevels - 1));
        };

        return digit2unit(cell2digit(pos_cell)) - digit2unit(cell2digit(neg_cell));
    }

    int ed_arrY, ed_arrColId;
    int ed_arrX, ed_arrRowId;
    int st_arrX, st_arrRowId;
    int st_arrY, st_arrColId;
    int lenx, leny;

    getRowPos(row, st_arrX, st_arrRowId);
    getRowPos(ed_row, ed_arrX, ed_arrRowId);

    getColPos(col, st_arrY, st_arrColId);
    getColPos(ed_col, ed_arrY, ed_arrColId);

    leny = ed_arrY - st_arrY + 1;
    lenx = ed_arrX - st_arrX + 1;
    at::Tensor data_cell = torch::empty({m, conf->cellsPerUnit * n}, op.dtype(torch::kI32));
    at::parallel_for(0, leny * lenx, 0, [&](int st, int ed) -> void {
        for (int k = st; k < ed; ++k)
        {
            int i = k / leny;
            int j = k % leny;
            int X, Y, x, y, m, n;

            if (i == 0)
                X = 0, x = st_arrRowId, m = lenx == 1 ? ed_arrRowId - st_arrRowId : conf->phyArrRowSize - st_arrRowId;
            else if (i == lenx - 1)
                X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = ed_arrRowId;
            else
                X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = conf->phyArrRowSize;

            if (j == 0)
                Y = 0, y = st_arrColId, n = leny == 1 ? ed_arrColId - st_arrColId : conf->usedCellsPerPhyRow - st_arrColId;
            else if (j == leny - 1)
                Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = ed_arrColId;
            else
                Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = conf->usedCellsPerPhyRow;
            if (m == 0 || n == 0)
                continue;
            data_cell.index({Slice(X, X + m), Slice(Y, Y + n)}) =
                phyArrManPro[arr[st_arrX + i][st_arrY + j]].readMat(x, y, m, n);
        }
    });

    auto cell2digit = [&](const at::Tensor &in) -> at::Tensor {
        at::Tensor data = torch::zeros({m, n}, op.dtype(torch::kI32));

        for (int i = 0; i < conf->cellsPerUnit; ++i)
        {
            data.add_(in.index({Slice(), Slice(i, n * conf->cellsPerUnit, conf->cellsPerUnit)}).__lshift__(i * conf->cellBits));
        }

        return data;
    };

    auto digit2unit = [&](const at::Tensor &in) -> at::Tensor {
        return in.to(torch::kFloat64).div((conf->unitLevels - 1) / 2.0).subtract(1).mul(conf->max_weight_value);
    };
    return digit2unit(cell2digit(data_cell));
}

/**
 * @param mat: 2d tensor
 * @param row: start row
 * @param col: start col
 */
void pimArrayPro::write_mat(const at::Tensor &matin, int row, int col)
{
    at::Tensor mat = matin.detach();
    int64_t m = mat.size(0), n = mat.size(1);
    int ed_row = std::min(row + m, rowSize);
    int ed_col = std::min(col + n, colSize);

    phyArrManPro.op_add(m*n, 0);
    auto getColPos = [&](int col, int &Y, int &arrCol) -> void {
        Y = col / conf->unitsPerPhyRow;
        arrCol = col % conf->unitsPerPhyRow * conf->cellsPerUnit;
    };

    auto getRowPos = [&](int row, int &X, int &arrRow) -> void {
        X = row / conf->phyArrRowSize;
        arrRow = row % conf->phyArrRowSize;
    };
    // getRowPos(ed_row, ed_arrX, ed_arrRowId);
    if (conf->mode == 0) // positive array & negative array
    {
        auto unit2digit = [&](const at::Tensor &x, at::Tensor &pos, at::Tensor &neg) -> void {
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
        auto digit2cell = [&](at::Tensor &in, at::Tensor &out) -> void {
            out = torch::cat({std::vector<at::Tensor>(conf->cellsPerUnit, in.view({m, n, 1}))}, 2);
            out.__irshift__(conf->crshift.to(in.device())).bitwise_and_(mask);
            out = out.view({m, n*conf->cellsPerUnit});
        };
        at::Tensor neg, pos, pos_cell, neg_cell;

        unit2digit(mat, pos, neg); //pos {}
        digit2cell(pos, pos_cell);
        digit2cell(neg, neg_cell);

        int ed_arrY, ed_arrColId;
        int ed_arrX, ed_arrRowId;
        int st_arrX, st_arrRowId;
        int st_arrY, st_arrColId;
        int lenx, leny;

        getRowPos(row, st_arrX, st_arrRowId);
        getRowPos(ed_row, ed_arrX, ed_arrRowId);

        getColPos(col, st_arrY, st_arrColId);
        getColPos(ed_col, ed_arrY, ed_arrColId);

        leny = ed_arrY - st_arrY + 1;
        lenx = ed_arrX - st_arrX + 1;
        if (conf->latency.enable)
        {
            int64_t Wr_num = 2*(int64_t)leny*lenx;
            double ti = conf->latency.parWrPhyNum <= 0 ? 1 : ceil(1.0*Wr_num / conf->latency.parWrPhyNum);
            phyArrManPro.latency_add(ti * conf->latency.latencyWrSinglePhyArr, 1);
        }

        at::parallel_for(0, leny * lenx, 0, [&](int st, int ed) -> void {
            for (int k = st; k < ed; ++k)
            {
                int i = k / leny;
                int j = k % leny;
                int X, Y, x, y, m, n;

                if (i == 0)
                    X = 0, x = st_arrRowId, m = lenx == 1 ? ed_arrRowId - st_arrRowId : conf->phyArrRowSize - st_arrRowId;
                else if (i == lenx - 1)
                    X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = ed_arrRowId;
                else
                    X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = conf->phyArrRowSize;

                if (j == 0)
                    Y = 0, y = st_arrColId, n = leny == 1 ? ed_arrColId - st_arrColId : conf->usedCellsPerPhyRow - st_arrColId;
                else if (j == leny - 1)
                    Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = ed_arrColId;
                else
                    Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = conf->usedCellsPerPhyRow;
                if (m == 0 || n == 0)
                    continue;
                phyArrManPro[arr[st_arrX + i][st_arrY + j]].writeMat(pos_cell.index({Slice(X, X + m), Slice(Y, Y + n)}), x, y);
                phyArrManPro[narr[st_arrX + i][st_arrY + j]].writeMat(neg_cell.index({Slice(X, X + m), Slice(Y, Y + n)}), x, y);
            }
        });

        return;
    }

    //ref column mode
    auto unit2digit = [&](const at::Tensor &x) -> at::Tensor {
        auto y = x.div(conf->max_weight_value);
        y.index_put_({y > 1}, 1.0);
        y.index_put_({y < -1}, -1.0);
        return y.add(1).div(2.0).mul(conf->unitLevels - 1).add(0.5).to(torch::kInt32);
    };

    int mask = (conf->cellLevels) - 1;
    auto digit2cell = [&](at::Tensor &in, at::Tensor &out) -> void {
        out = torch::cat({std::vector<at::Tensor>(conf->cellsPerUnit, in.view({m, n, 1}))}, 2);
        out.__irshift__(conf->crshift.to(in.device())).bitwise_and_(mask);
        out = out.view({m, -1});
    };

    at::Tensor data = unit2digit(mat);
    at::Tensor data_cell;
    digit2cell(data, data_cell);

    int ed_arrY, ed_arrColId;
    int ed_arrX, ed_arrRowId;
    int st_arrX, st_arrRowId;
    int st_arrY, st_arrColId;
    int lenx, leny;

    getRowPos(row, st_arrX, st_arrRowId);
    getRowPos(ed_row, ed_arrX, ed_arrRowId);

    getColPos(col, st_arrY, st_arrColId);
    getColPos(ed_col, ed_arrY, ed_arrColId);

    leny = ed_arrY - st_arrY + 1;
    lenx = ed_arrX - st_arrX + 1;
    if (conf->latency.enable)
    {
        int64_t Wr_num = (int64_t)leny*lenx;
        double ti = conf->latency.parWrPhyNum<=0? 1 : ceil(1.0*Wr_num/conf->latency.parWrPhyNum);
        phyArrManPro.latency_add(ti*conf->latency.latencyWrSinglePhyArr, 1);
    }
    at::parallel_for(0, leny * lenx, 0, [&](int st, int ed) -> void {
        for (int k = st; k < ed; ++k)
        {
            int i = k / leny;
            int j = k % leny;
            int X, Y, x, y, m, n;

            if (i == 0)
                X = 0, x = st_arrRowId, m = lenx == 1 ? ed_arrRowId - st_arrRowId : conf->phyArrRowSize - st_arrRowId;
            else if (i == lenx - 1)
                X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = ed_arrRowId;
            else
                X = i * conf->phyArrRowSize - st_arrRowId, x = 0, m = conf->phyArrRowSize;

            if (j == 0)
                Y = 0, y = st_arrColId, n = leny == 1 ? ed_arrColId - st_arrColId : conf->usedCellsPerPhyRow - st_arrColId;
            else if (j == leny - 1)
                Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = ed_arrColId;
            else
                Y = j * conf->usedCellsPerPhyRow - st_arrColId, y = 0, n = conf->usedCellsPerPhyRow;
            if (m == 0 || n == 0)
                continue;
            phyArrManPro[arr[st_arrX + i][st_arrY + j]].writeMat(data_cell.index({Slice(X, X + m), Slice(Y, Y + n)}), x, y);
        }
    });
}

/**
 * Performs mm.
* @param mat input matrix, should be 2D or 3D Tensor of sizes {batch_size, rowSize} or {batch_size, row_size, col_size}
* @return 2d tensor of sizes {batch_size, colSize}
*/
at::Tensor pimArrayPro::mm(const at::Tensor &matin)
{
    at::Tensor mat = matin.detach();

    auto mm2d = [&](const at::Tensor &mat) -> at::Tensor {
        int batch_size = mat.size(0);
        int siz = mat.size(1);
        double max_one;
        phyArrManPro.op_add(  ((int64_t)rowSize*colSize + (int64_t)(rowSize-1)*colSize)*batch_size, 1);
        at::Tensor input = phyArrayPro::preWorkForMM(mat, conf, std::ref(max_one)); // input should be tensor of size {batch_size, inBits/inVBits, rowSize}

        at::Tensor out = torch::zeros({arrX_size, batch_size, arrY_size * conf->unitsPerPhyRow}, op.dtype(torch::kF64));

        // postive & negative array mode
        if (conf->mode == 0)
        {
            if (conf->energy.enable)
            {
                phyArrManPro.add_adder_energy(2 * (arrY_size - 1) * arrY_size * phyArrManPro.access(0).colSize * conf->energy.adderEnergy);
            }
            at::Tensor nout = torch::zeros({arrX_size, batch_size, arrY_size * conf->unitsPerPhyRow}, op.dtype(torch::kF64));
            at::parallel_for(0, arrX_size * arrY_size, 0, [&](int st, int ed) {
                for (int k = st; k < ed; ++k)
                {
                    int i = k / arrY_size;
                    int j = k % arrY_size;
                    if (i * conf->phyArrRowSize >= siz)
                        continue;
                    out[i].index({Slice(), Slice(j * conf->unitsPerPhyRow, (j + 1) * conf->unitsPerPhyRow)}) = phyArrManPro[arr[i][j]].mm(input.index({Slice(), Slice(), Slice(i * conf->phyArrRowSize, (i + 1) * conf->phyArrRowSize)}), conf, max_one, phyArrayPro::postWorkForMM);

                    nout[i].index({Slice(), Slice(j * conf->unitsPerPhyRow, (j + 1) * conf->unitsPerPhyRow)}) = phyArrManPro[narr[i][j]].mm(input.index({Slice(), Slice(), Slice(i * conf->phyArrRowSize, (i + 1) * conf->phyArrRowSize)}), conf, max_one, phyArrayPro::postWorkForMM);
                }
            });
            out.subtract_(nout);
        }
        else // ref col mode
        {
            if (conf->energy.enable)
            {
                phyArrManPro.add_adder_energy((arrY_size - 1) * arrY_size * phyArrManPro.access(0).colSize * conf->energy.adderEnergy);
            }
            at::parallel_for(0, arrX_size * arrY_size, 0, [&](int st, int ed) {
                for (int k = st; k < ed; ++k)
                {
                    int i = k / arrY_size;
                    int j = k % arrY_size;
                    if (i * conf->phyArrRowSize >= siz)
                        continue;
                    out[i].index({Slice(), Slice(j * conf->unitsPerPhyRow, (j + 1) * conf->unitsPerPhyRow)}) = phyArrManPro[arr[i][j]].mm(input.index({Slice(), Slice(), Slice(i * conf->phyArrRowSize, (i + 1) * conf->phyArrRowSize)}), conf, max_one, phyArrayPro::postWorkForMM);
                }
            });
        }
        return out.sum(0).index({Slice(), Slice(0, colSize)});
    };

    if (mat.sizes().size() == 2)
    {
        if (conf->latency.enable)
        {
            double ti = conf->latency.phyMMLatency* (conf->latency.parMMPhyNum<=0? 1 : ceil(1.0*arrX_size*arrY_size/conf->latency.parMMPhyNum));
            phyArrManPro.latency_add(conf->inPluses*ti + ceil(1.0*(arrX_size-1)/(conf->latency.addTreeWideSize-1))*conf->latency.addTreeLatency*ceil(1.0*arrY_size/conf->latency.addTreeSharedNum), 0);
        }
        return mm2d(mat);
    }
    else
    {            
        int batch_size, row_size, col_size;
        batch_size = mat.size(0);
        row_size = mat.size(1);
        col_size = mat.size(2);
        if (conf->latency.enable)
        {
            double ti = conf->latency.phyMMLatency* (conf->latency.parMMPhyNum<=0? 1 : ceil(1.0*arrX_size*arrY_size/conf->latency.parMMPhyNum));
            phyArrManPro.latency_add(batch_size*(conf->inPluses*ti + ceil(1.0*(arrX_size-1)/(conf->latency.addTreeWideSize-1))*conf->latency.addTreeLatency*ceil(1.0*arrY_size/conf->latency.addTreeSharedNum)), 0);
        }
        return mm2d(mat.reshape({-1, col_size})).reshape({batch_size, row_size, -1});
    }
}

#endif
