#ifndef PIMTORCH_PHYARRAYSIMPLE_H
#define PIMTORCH_PHYARRAYSIMPLE_H

#include <torch/torch.h>
#include <torch/custom_class.h>
#include <iostream>
#include "ir_drop_solve.h"


using std::cout;
using std::endl;
using torch::TensorOptions;
using namespace torch::indexing;

struct phyArrayPro
{
    phyArrayPro(int rowSize, int colSize, torch::TensorOptions op = {}, const pim_array_pro_config *conf = &pro_decf()): conf(conf)
    {
        this->rowSize = rowSize;
        this->colSize = colSize;
        readEnergy = writeEnergy = computeEnergy = 0;
        deltaConduct = (conf->maxConduct - conf->minConduct) / (conf->cellLevels-1);
        this->op = op;
        if (!conf->C2C_en)
        {
            if (conf->cellBits <= 8)
                data = torch::zeros({rowSize, colSize}, op.dtype(torch::kUInt8));
            else if (conf->cellBits < 32)
                data = torch::zeros({rowSize, colSize}, op.dtype(torch::kInt32));
            else
            {
                throw "too big cellBits, in phyArrayPro";
            }
        }
        else
        {
            data = torch::zeros({rowSize, colSize}, op.dtype(torch::kF32));
        }
        
    
        totalWrCnt = 0;

        if (conf->write_cnt_en)
        {
            cellWrCnt = torch::zeros({rowSize, colSize}, op.dtype(torch::kInt64));
            totalCmpWrCnt = 0;
        }
        if (conf->sa.enable)
        {
            p_state = torch::empty({rowSize, colSize}, op.dtype(torch::kF32));
            p_state.uniform_();
            p_state_index_sa0 = p_state.le(conf->sa.init_pSA0);
            p_state_index_sa1 = p_state.gt(1-conf->sa.init_pSA1);
            data.index_put_({p_state_index_sa1}, conf->cellLevels-1);
        }
    }

    void writeMat(const at::Tensor &mat, int row = 0, int col = 0);

    at::Tensor readMat(int row, int col, int m = index_len_max, int n = index_len_max);

    template <typename F, typename Fdata, typename T, typename DataToPostWork>
    at::Tensor mm(const at::Tensor &mat, F preWork, Fdata pim_conf, DataToPostWork d, T postWork)
    {
        return postWork(mm(preWork(mat, pim_conf, std::ref(d))), pim_conf, std::ref(d));
    }

    template <typename T, typename Fdata, typename DataToPostWork>
    at::Tensor mm(const at::Tensor &mat, Fdata pim_conf, DataToPostWork d, T postWork)
    {
        return postWork(mm(mat), pim_conf, std::ref(d));
    }

    at::Tensor mm(const at::Tensor &mat);

    void print(std::ostream &);

    static at::Tensor preWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one);
    static at::Tensor postWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one);
    static const int index_len_max = std::numeric_limits<int>::max() / 2;

    const pim_array_pro_config *const conf;
    int rowSize, colSize;
    double deltaConduct;
    double readEnergy, writeEnergy, computeEnergy;
    // double area; not support area now
    // int64_t latency; not support latency now

    torch::TensorOptions op;
    int64_t totalWrCnt, totalCmpWrCnt;

    at::Tensor cellWrCnt;
    at::Tensor data;
    at::Tensor p_state;
    at::Tensor p_state_index_sa0;
    at::Tensor p_state_index_sa1;
};

/**
 * Performs write to the phy array.
* @param mat input matrix, should be 2D Tensor
* @param row write matrix from [row, col] to [row+mat.size(0)-1, col+mat.size(1)-1]
* @param col write matrix from [row, col] to [row+mat.size(0)-1, col+mat.size(1)-1]
* @return void
*/
void phyArrayPro::writeMat(const at::Tensor &matin, int row, int col)
{
    auto mat = matin;

    int64_t m = mat.size(0), n = mat.size(1);
    totalWrCnt += m * n;
    if (conf->sa.enable && conf->sa.runtime_en)
    {
        at::Tensor area = torch::zeros({rowSize, colSize}, op.dtype(torch::kBool)).index_put_({Slice(row, row+m), Slice(col, col+n)}, true);
        p_state.uniform_();

        p_state_index_sa0.logical_or_(p_state.le(conf->sa.pSA0).logical_and(p_state_index_sa1.logical_not()).logical_and(area));
        p_state_index_sa1.logical_or_(p_state.gt(1-conf->sa.pSA1).logical_and(p_state_index_sa0.logical_not()).logical_and(area));

    }
    if (conf->sa.enable)
    {
        mat.index_put_({p_state_index_sa1.index({Slice(row, row+m), Slice(col, col+n)})}, conf->cellLevels-1);
        mat.index_put_({p_state_index_sa0.index({Slice(row, row+m), Slice(col, col+n)})}, 0);
    }
    if (conf->write_cnt_en)
    {
        //currently, cell write cnt is simply equal to whther it is wrriten or not, does not based on pluse number when cell bits >1
        at::Tensor add = (data.index({Slice(row, row + m), Slice(col, col + n)}) != mat);
        cellWrCnt.index({Slice(row, row + m), Slice(col, col + n)}).add_(add);

        totalCmpWrCnt += add.sum().item<int64_t>();
    }

    // remains future works
    if (conf->energy.enable)
    {
        at::Tensor diff = data.index({Slice(row, row + m), Slice(col, col + n)}) != mat;
        int64_t cmpWriteNum = diff.sum().item<int64_t>();
        double energy = cmpWriteNum * conf->energy.averageEnergyPerWrite;
        writeEnergy += energy;
//        if (conf->wm == phy_array_writeMode::V_Div_2)
//        {
//            double energy = conf->writeV/2 * conf->writeV/2 * conf->lat_area.phyWrLatency/(conf->cellLevels - 1);
//            double cnt = 0;
//
//            if (conf->energy.writeUseProbability)
//            {
//                double average_conductance = 0;
//
//                for (int i = 0; i < conf->energy.CellPD.size(); ++i)
//                {
//                    average_conductance += i * conf->energy.CellPD[i];
//                }
//                average_conductance *= deltaConduct;
//                average_conductance += conf->minConduct;
//
//                cnt = m * n * ((rowSize + colSize - 2) * average_conductance + average_conductance * 4);
//
//                double average_delta = data.index({Slice(row, row + m), Slice(col, col + n)}).sub(mat)
//                        .abs().sum().div(m * n).item<double>();
//                cnt *= average_delta;
//            }
//            else
//            {
//                for (int i = 0; i < m; ++i)
//                {
//                    for (int j = 0; j < n; ++j)
//                    {
//                        cnt += ((data.index({row + i, Slice(0, col)}).sum().item<double>()
//                                 + mat.index({i, Slice(0, j)}).sum().item<double>()
//                                 + data.index({row + i, Slice(col + j + 1, colSize)}).sum().item<double>()
//                                 + data.index({Slice(0, row), col + j}).sum().item<double>()
//                                 + mat.index({Slice(0, i), j}).sum().item<double>()
//                                 + data.index({Slice(row + i + 1, rowSize), col + j}).sum().item<double>()
//                                 + (data[row + i][col + j].item<double>() + mat[i][j].item<double>())/2 * 4)
//                                * deltaConduct + (rowSize + colSize - 1) * conf->minConduct)
//                               * fabs(data[row + i][col + j].item<double>() - mat[i][j].item<double>());
//                    }
//                }
//            }
//
//            energy *= cnt;
//            //add circuit energy
//            energy += m * n * (rowSize * conf->energy.writeRowPeripheryEnergy
//                               + colSize * conf->energy.writeColPeripheryEnergy);
//
//            writeEnergy += energy;
//        }
//        else
//        {
//            std::cout << "we now don't support other write mode!" << std::endl;
//        }
//
    }

    if (conf->C2C_en)
    {
        //1/(delta*data + minConduct) = Rtarget * exp(q);  q ~ N(0, theta)   
        // data = (exp(-q) * Ctarget - minConduct)/deltaConduct;   
        at::Tensor nol = torch::normal(0, conf->C2C_theta, {m, n}, {}, op.dtype(torch::kF32)).mul(-1).exp(); 
        at::Tensor wr = ((mat.to(torch::kF32)*deltaConduct+conf->minConduct)*nol-conf->minConduct)/deltaConduct;

        data.index_put_({Slice(row, row + m), Slice(col, col + n)}, wr);
        return;
    }
    data.index_put_({Slice(row, row + m), Slice(col, col + n)}, mat);
}

/**
 * Performs read to the phy array. 
* read matrix: from [row, col] to [row+m, col+n]
* @param row read matrix start row
* @param col read matrix start col
* @param m read matrix row Size
* @param n read matrix colSize
* @return 2D Tensor sizes = [m, n],
*/
at::Tensor phyArrayPro::readMat(int row, int col, int m, int n)
{
    if (conf->energy.enable)
    {
        if (conf->rm == phy_array_readMode::Ground)
        {
            double energy = conf->readV * conf->readV * conf->lat_area.phyRdLatency;
            int elementNum = m * colSize;
            double conductance;

            if (conf->energy.readUseProbability)
            {
                double average = 0;

                for (int i = 0; i < conf->energy.CellPD.size(); ++i)
                {
                    average += i * conf->energy.CellPD[i];
                }

                conductance = elementNum * average * deltaConduct;
            }
            else
            {
                conductance = data.index({Slice(row, row + m)}).sum().item<double>() * deltaConduct;
            }

            conductance += elementNum * conf->minConduct;
            energy *= conductance;//array energy
            //add circuit energy
            energy += m * conf->energy.readRowPeripheryEnergy + colSize * conf->energy.readColPeripheryEnergy;
            readEnergy += energy;
        }
        else
        {
            std::cout << "we now don't support other read mode!" << std::endl;
        }

    }
    if (conf->C2C_en)
        return data.round().to(torch::kInt32).index({Slice(row, row + m), Slice(col, col + n)});
    return data.index({Slice(row, row + m), Slice(col, col + n)});
}

/**
 * Performs matrix mul matrix.
* @param mat input matrix, {batch_size, inBits/VinBits, phy_rowSize}, data range from 0 to 1.
* @return mat x data, 3d Tensor. {batch_size, inBits/VinBits, phy_colSize}, range from 0 to 1
*/
at::Tensor phyArrayPro::mm(const at::Tensor &mat)
{
    // remains future works
    if (conf->energy.enable)
    {
        double energy = conf->computeV * conf->computeV * conf->lat_area.phyMMLatency;
        if (conf->energy.computeUseProbability)
        {
            double average_conductance = 0, average_V_square = 0;

            for (int i = 0; i < conf->energy.CellPD.size(); ++i)
            {
                average_conductance += i * conf->energy.CellPD[i];
            }
            average_conductance *= deltaConduct;
            average_conductance += conf->minConduct;

            for (int i = 0; i < conf->energy.inVPD.size(); ++i)
            {
                average_V_square += i * i * conf->energy.inVPD[i];
            }
            average_V_square /= conf->inVLevels * conf->inVLevels;

            //array energy
            energy *= average_V_square * average_conductance * mat.size(2) * colSize * mat.size(0) * mat.size(1);
        }
        else
        {
            //array energy
           // cout << mat.sizes() << endl;
            //cout << data.index({Slice(0, mat.size(2))}).sizes() << endl;
            auto G_matrix = data.index({Slice(0, mat.size(2))}).to(torch::kF64)
                    .mul(deltaConduct).add(conf->minConduct);
            energy *= mat.pow(2).matmul(G_matrix).sum().item<double>();
        }

        //add circuit energy
        energy += (mat.size(2) * (conf->energy.computeRowPeripheryEnergy + conf->energy.DACEnergy) + colSize * (conf->energy.computeColPeripheryEnergy + conf->energy.ADCEnergy))* mat.size(0) * mat.size(1);
        computeEnergy += energy;
        //todo:
    }
    if (conf->ir_drop.enable)
    {
        auto matin = mat;
        if (mat.size(2)<rowSize)
        {
            matin = torch::zeros({mat.size(0), mat.size(1), rowSize}, mat.options());
            matin.index({Slice(), Slice(), Slice(0, mat.size(2))}) = mat;
        }
        if (conf->ir_drop.fast_mode)
        {
            return ir_drop_solve_fast(matin*conf->computeV, data.to(torch::kF64) * deltaConduct + conf->minConduct, rowSize, colSize, conf->ir_drop.times).div(rowSize*conf->maxConduct*conf->computeV);
        }
        else
            return ir_drop_solve_acc(matin*conf->computeV, data.to(torch::kF64) * deltaConduct + conf->minConduct, rowSize, colSize).div(rowSize*conf->maxConduct*conf->computeV);

    }
    return torch::matmul(mat, data.index({Slice(0, mat.size(2))}).to(torch::kF64) * deltaConduct + conf->minConduct).div(rowSize*conf->maxConduct);
}

/**
 * Before you use mm. you need to call preWorkForMM first for pretreatment.
* mat is the data you want to perform mm. e.g. [3 0 4], but you should reverts it to voltage vector (might be multi pulses, so 1d tensor mat will return 2d tensor).
* @param mat input matrix. {batch_size, rowSize}
* @return Tensor. return 3d tensor {batch_size, inBits/inVBits, rowSize}.
*/
at::Tensor phyArrayPro::preWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one)
{
    if (conf->dynamic_max_input)
    {
        if (conf->has_negative_input)
            max_one = std::min(conf->max_phy_input_value, mat.abs().max().item<double>());
        else
            max_one = std::min(conf->max_phy_input_value, mat.max().item<double>());
        // std::cout << "max_one = " << max_one << std::endl;
    }
    else
        max_one = conf->max_phy_input_value;

    at::Tensor out;

    // int nums = conf->inBits / conf->inVBits;
    double k = (conf->inLevels - 1) / max_one;

    if (conf->has_negative_input)
    {
        if (conf->inBits <= 1)
        {
            throw("has neg input, inbits should be >1.");
        }
        auto index_larger = mat > max_one;
        auto index_smaller = mat < -max_one;

        if (!conf->trunc_input)
        {
            int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
            int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();

            if (num_larger_than_max + num_smaller_than_min != 0)
            {
                cout << "trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array." << endl;
                throw("trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array.");
            }
        }

        if (conf->inVBits == 1) // complement form to represent number
        {
            out = mat.add(max_one).mul(k / 2);
            out.index_put_({index_larger}, conf->inLevels - 1);
            out.index_put_({index_smaller}, 0);

            out = out.round_().to(torch::kInt32).bitwise_xor((1 << (conf->inBits - 1)));
            max_one *= 2;
        }
        else //support negative voltage for negative number
        {
            auto neg = mat < 0;
            
            out = mat.abs().mul_(k).round_().to(torch::kI32);
            out.index_put_({index_larger}, conf->inLevels - 1);
            out.index_put_({index_smaller}, conf->inLevels - 1);

            at::Tensor output = torch::empty({out.size(0), conf->inPluses, out.size(1)}, TensorOptions(mat.device()).dtype(torch::kFloat64));
            int mask = conf->inVLevels - 1;
            neg = neg.view({out.size(0), 1, out.size(1)});
            auto neg_index = torch::cat({std::vector<at::Tensor>(conf->inPluses, neg)}, 1);

            out = out.view({out.size(0), out.size(1), 1});
            out = torch::cat({std::vector<at::Tensor>(conf->inPluses, out)}, 2);
            output.index({Ellipsis}) = out.__irshift__(conf->rshift.to(mat.device())).bitwise_and_(mask).div((double)(conf->inVLevels - 1)).transpose_(1, 2);
            /*at::parallel_for(0, conf->inPluses, 0, [&](int st, int ed) {
                for (int i = st; i < ed; ++i)
                {
                    output.index_put_({Slice(), Slice(i, i + 1)}, (out.__rshift__(i * conf->inVBits).bitwise_and(mask).div((double)(conf->inVLevels - 1))));
                    neg_index.index_put_({Slice(), Slice(i, i + 1)}, neg);
                }
            });*/
            return output.index_put({neg_index}, output.index({neg_index})*-1 );
        }
    }
    else // input vector should be positive
    {
        auto index_larger = mat > max_one;
        auto index_smaller = mat < 0;

        if (!conf->trunc_input)
        {
            int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
            int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();
            //check if there exists values that are larger (or smaller) than max_one (or 0).
            if (num_larger_than_max + num_smaller_than_min != 0)
            {
                cout << "trunc_input = false, && input exsits value larger than maximum or less than min. please modify config of array." << endl;
                throw("trunc_input = false, && input exsits value larger than maximum or less than min. please modify config of array.");
            }
            out = mat.mul_(k).round_().to(torch::kInt32);
        }
        else
        {
            out = mat.mul_(k).round_().to(torch::kInt32);
            out.index_put_({index_larger}, conf->inLevels - 1);
            out.index_put_({index_smaller}, 0);
        }
        //now out is size of [batch_size, rowSize], type is int
    }

    // out is {batch_size, rowSize}
    at::Tensor output = torch::empty({out.size(0), conf->inPluses, out.size(1)}, TensorOptions(mat.device()).dtype(torch::kFloat64));
    int mask = conf->inVLevels - 1;
    out = out.view({out.size(0), out.size(1), 1});
    out = torch::cat({std::vector<at::Tensor>(conf->inPluses, out)}, 2);
    output.index({Ellipsis}) = out.__irshift__(conf->rshift.to(mat.device())).bitwise_and_(mask).div((double)(conf->inVLevels -1)).transpose_(1, 2);
    /*at::parallel_for(0, conf->inPluses, 0, [&](int st, int ed) {
        for (int i = st; i < ed; ++i)
        {
            output.index_put_({Slice(), Slice(i, i + 1)}, (out.__rshift__(i * conf->inVBits).bitwise_and(mask).div((double)(conf->inVLevels - 1))) );
        }
    });*/

    return output;
}

/**
* @param mat: sizes {batch_size, inBits/inVbits, phyColSize}
* @param post: data will be used in post work
* @param max_one: data will be used in post work, max_one is send from prework
* @return tensor sizes {batch_size, unitNumPerPhyRow}
*/
at::Tensor phyArrayPro::postWorkForMM(const at::Tensor &mat, const pim_array_pro_config *conf, double &max_one)
{
    // int nums = conf->inBits/conf->inVBits;
    double scalar_value = max_one *conf->phyArrRowSize /(conf->unitLevels-1) * conf->max_weight_value /(conf->outLevels -1) *(conf->inVLevels-1) / (conf->inLevels-1) * (conf->cellLevels-1);
    at::Tensor out;

    if (conf->mode == 1) // ref col mode
    {
        scalar_value *= conf->maxConduct/(conf->maxConduct-conf->minConduct);
        out = mat.mul(conf->outLevels - 1).round_(); //let out range from 0 -- outLevels -1, double

        out.index({Ellipsis, Slice(0, conf->phyArrColSize+1)}).subtract_(out.index({Ellipsis, Slice(conf->phyArrColSize+1)}));
        
        at::Tensor refValue = out.index({Ellipsis, Slice(conf->phyArrColSize, conf->phyArrColSize+1 ) });

        refValue = refValue.__lshift__(conf->unitBits).subtract_(refValue);

        out = out.index({Slice(), Slice(), Slice(0, conf->usedCellsPerPhyRow)});

        at::Tensor output = torch::empty({mat.size(0), conf->inPluses, conf->unitsPerPhyRow}, mat.device());
        // at::Tensor unitScalar = torch::ones({nums, conf->usedCellsPerPhyRow}, TensorOptions(mat.device()).dtype(torch::kI32));

        // at::parallel_for(0, conf->cellsPerUnit, 0, [&](int st, int ed) -> void {
        //     for (int i = st; i < ed; ++i)
        //     {
        //         unitScalar.index({Slice(), Slice(i, conf->usedCellsPerPhyRow, conf->cellsPerUnit)}).__ilshift__(i*conf->cellBits);
        //     }
        // });

        
        out *= conf->unitScalar.to(mat.device());
        
        output.index({Ellipsis}) = out.transpose_(2, 1).view({mat.size(0), conf->unitsPerPhyRow, conf->cellsPerUnit, conf->inPluses}).sum(2).transpose_(1, 2);
//        at::parallel_for(0, conf->unitsPerPhyRow, 0, [&](int st, int ed) -> void {
//            for (int i = st; i < ed; ++i)
//            {
//
//                output.index_put_({Slice(), Slice(), Slice(i, i+1)}, out.index({Slice(), Slice(), Slice(i*conf->cellsPerUnit, (i+1)*conf->cellsPerUnit)}).sum(2, true));
//            }
//        });

        output = output.__lshift__(1)-refValue;

        // at::Tensor inScalar = torch::ones({nums, 1}, TensorOptions(mat.device()).dtype(torch::kF64));
        
        // at::parallel_for(0, nums, 0, [&](int st, int ed) -> void {
        //     for (int i = st; i < ed; ++i)
        //     {
        //         inScalar.index_put_({Slice(i, i + 1)}, 1 << (i*conf->inVBits));
        //     }
        // });
        // if (conf->inVBits==1 && conf->has_negative_input)
        //     inScalar.index_put_({Slice(conf->inBits-1)}, (1 << (conf->inBits-1))*-1);

        
        out = (output* conf->inScalar.to(mat.device())).sum(1).mul_(scalar_value);  //{batch_size, unitNumPerPhyRow}
        return out;
    }
    else // postive & negative array mode,  in this single array, normal calculation.
    {

        out = mat.mul(conf->outLevels - 1).round_().index({Ellipsis, Slice(0, conf->usedCellsPerPhyRow)}); //let out range from 0 -- outLevels -1, double
        at::Tensor output = torch::empty({mat.size(0), conf->inPluses, conf->unitsPerPhyRow}, mat.device());
        // at::Tensor unitScalar = torch::ones({nums, conf->usedCellsPerPhyRow}, TensorOptions(mat.device()).dtype(torch::kI32));

        // at::parallel_for(0, conf->cellsPerUnit, 0, [&](int st, int ed) -> void {
        //     for (int i = st; i < ed; ++i)
        //     {
        //         unitScalar.index({Slice(), Slice(i, conf->usedCellsPerPhyRow, conf->cellsPerUnit)}).__ilshift__(i*conf->cellBits);
        //     }
        // });

        out *= conf->unitScalar.to(mat.device());
        
        output.index({Ellipsis}) = out.transpose_(2, 1).view({mat.size(0), conf->unitsPerPhyRow, conf->cellsPerUnit, conf->inPluses}).sum(2).transpose_(1, 2);
//        at::parallel_for(0, conf->unitsPerPhyRow, 0, [&](int st, int ed) -> void {
//            for (int i = st; i < ed; ++i)
//            {
//                output.index_put_({Slice(), Slice(), Slice(i, i+1)}, out.index({Slice(), Slice(), Slice(i*conf->cellsPerUnit, (i+1)*conf->cellsPerUnit)}).sum(2, true));
//            }
//        });

        // at::Tensor inScalar = torch::ones({nums, 1}, TensorOptions(mat.device()).dtype(torch::kF64));

        // at::parallel_for(0, nums, 0, [&](int st, int ed) -> void {
        //     for (int i = st; i < ed; ++i)
        //     {
        //         inScalar.index_put_({Slice(i, i + 1)}, 1 << (i*conf->inVBits));
        //     }
        // });

        out = (output * conf->inScalar.to(mat.device())).sum(1).mul_(scalar_value); //{batch_size, unitNumPerPhyRow}
        return out;
    }
}

void phyArrayPro::print(std::ostream &os)
{
    using std::endl;
    os << "data = " << endl;
    os << data << endl;
    if (conf->energy.enable)
    {
        //todo:
//        os << " total energy = " << readEnergy + writeEnergy
//           << ", write energy = " << writeEnergy
//           << ", read energy = " << readEnergy << endl;
    }
    os << " total write cnt = " << totalWrCnt << endl;
    if (conf->write_cnt_en)
    {
        os << " total cmp write cnt = " << totalCmpWrCnt << endl;
        os << "cell write cnt = " << endl;
        os << cellWrCnt << endl;
    }
}

#endif //PIMTORCH_PHYARRAYSIMPLE_H
