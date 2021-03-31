#ifndef PIMTORCH_PHYARRAYSIMPLE_H
#define PIMTORCH_PHYARRAYSIMPLE_H

#include <torch/torch.h>
#include <torch/custom_class.h>
#include <iostream>

using std::cout;
using std::endl;
using torch::TensorOptions;
using torch::indexing::Slice;

/**
 * phyArraySimple implements a phy NVM crossbar array with size [rowSize, colSize]
 * The cell in array is digit. Only has minConduct and maxConduct two states.
 */
struct phyArraySimple
{
    phyArraySimple(int rowSize = 128, int colSize = 128, bool toGPU = false) : rowSize(rowSize), colSize(colSize)
    {
        double minConduct = 1e-6; //define minimum & maximum conduct of a cell
        double maxConduct = 1e-4;
        double readVoltage = 0.5; //define read voltage

        ImaxPCell = readVoltage * maxConduct; //get current under read voltage
        IminPCell = readVoltage * minConduct;
        deltaI = ImaxPCell - IminPCell;
        totalCmpWrCnt = totalWrCnt = 0;

        device = toGPU ? torch::kCUDA : torch::kCPU;
        torch::TensorOptions op(device);
        data = at::full({rowSize, colSize}, IminPCell, op.dtype(torch::kFloat64)); //data in type of current, Imin or Imax
        //dataDigit = at::full({rowSize, colSize}, 0, op.dtype(torch::kInt8));         //data in type of digit value 0(Imin)/1(Imax)
        dataDigit = at::zeros({rowSize, colSize}, op.dtype(torch::kInt8)); //data in type of digit value 0(Imin)/1(Imax)
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

    void initWriteMat(const at::Tensor &in)
    {
        auto write_it = [&](const at::Tensor &in, int x, int y, int m, int n) -> void {
            //std::cout << in << std::endl;
            at::Tensor writeIn = at::full({m, n}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
            //std::cout << dataDigit.index({Slice(x, x+m), Slice(y, y+n)}) << std::endl;
            at::Tensor add = dataDigit.index({Slice(x, x + m), Slice(y, y + n)}).bitwise_xor(in.to(torch::kInt8));

            totalWrCnt += (m * n);
            totalCmpWrCnt += (add.sum().item<int64_t>());
            cellWrCnt.index_put_({Slice(x, x + m), Slice(y, y + n)}, cellWrCnt.index({Slice(x, x + m), Slice(y, y + n)}).add(add));

            writeIn.mul_(in).add_(IminPCell);
            data.index_put_({Slice(x, x + m), Slice(y, y + n)}, writeIn);
            dataDigit.index_put_({Slice(x, x + m), Slice(y, y + n)}, in);
        };
        write_it(in, 0, 0, rowSize, colSize);
    }
    /*
    *   m -> in.size(0), n -> in.size(1)
    *   write mat in to our array.
    * */
    void writeMat(const at::Tensor &in, int m, int n)
    {
        at::Tensor writeIn = at::full({m, n}, deltaI, TensorOptions(device).dtype(torch::kFloat64));
        at::Tensor add = dataDigit.index({Slice(0, m), Slice(0, n)}).bitwise_xor(in);

        totalWrCnt += (m * n);
        totalCmpWrCnt += add.sum().item<int64_t>();
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

struct phyArrayPro
{

    phyArrayPro(int rowSize, int colSize, torch::TensorOptions op = {}, phy_array_config conf = phy_decf)
    {
        this->rowSize = rowSize;
        this->colSize = colSize;
        conf = phy_decf;
        readEnergy = writeEnergy = 0;
        levels = 1 << conf.cellBits;
        deltaConduct = (conf.maxConduct - conf.minConduct) / levels;
        this->op = op;
        if (conf.cellBits <= 8)
            data = torch::zeros({rowSize, colSize}, op.dtype(torch::kUInt8));
        else if (conf.cellBits < 32)
            data = torch::zeros({rowSize, colSize}, op.dtype(torch::kInt32));
        else
        {
            throw "too big cellBits, in phyArrayPro";
        }

        // to calculate energy, we need write cnt
        if (conf.energy_cal_en)
            conf.write_cnt_en = true;

        totalWrCnt = 0;

        if (conf.write_cnt_en)
        {
            cellWrCnt = torch::zeros({rowSize, colSize}, op.dtype(torch::kInt64));
            totalCmpWrCnt = 0;
        }
    }

    void writeMat(const at::Tensor &mat, int row = 0, int col = 0);

    at::Tensor readMat(int row, int col, int m = index_len_max, int n = index_len_max);

    template <typename F, typename Fdata, typename T, typename DataToPostWork>
    at::Tensor mm(const at::Tensor &mat, F preWork, Fdata pim_conf, DataToPostWork d, T postWork)
    {
        return postWork(mm(preWork(mat, pim_conf, &conf, std::ref(d))), pim_conf, &conf, std::ref(d));
    }

    template <typename T, typename Fdata, typename DataToPostWork>
    at::Tensor mm(const at::Tensor &mat, Fdata pim_conf, DataToPostWork d, T postWork)
    {
        return postWork(mm(mat), pim_conf, &conf, std::ref(d));
    }

    at::Tensor mm(const at::Tensor &mat);

    void print(std::ostream &);

    static at::Tensor preWorkForMM(const at::Tensor &mat, pim_array_config *pconf, phy_array_config *conf, double &max_one);
    static at::Tensor postWorkForMM(const at::Tensor &mat, pim_array_config *pconf, phy_array_config *conf, double &max_one);
    static const int index_len_max = std::numeric_limits<int>::max() / 2;

    phy_array_config conf;
    int rowSize, colSize, levels;
    double deltaConduct;
    double readEnergy, writeEnergy;
    // double area; not support area now
    // int64_t latency; not support latency now

    torch::TensorOptions op;
    int64_t totalWrCnt, totalCmpWrCnt;

    at::Tensor cellWrCnt;
    at::Tensor data;
};

/**
 * Performs write to the phy array.
* @param mat input matrix, should be 2D Tensor
* @param row write matrix from [row, col] to [row+mat.size(0)-1, col+mat.size(1)-1]
* @param col write matrix from [row, col] to [row+mat.size(0)-1, col+mat.size(1)-1]
* @return void
*/
void phyArrayPro::writeMat(const at::Tensor &mat, int row, int col)
{

    int64_t m = mat.size(0), n = mat.size(1);
    totalWrCnt += m * n;
    if (conf.write_cnt_en)
    {
        //currently, cell write cnt is simply equal to whther it is wrriten or not, does not based on pluse number when cell bits >1
        at::Tensor add = (data.index({Slice(row, row + m), Slice(col, col + n)}) != mat);
        cellWrCnt.index({Slice(row, row + m), Slice(col, col + n)}).add_(add);
        totalCmpWrCnt += add.sum().item<int64_t>();
    }

    // remains future works
    if (conf.energy_cal_en)
    {
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
    // remains future works
    if (conf.energy_cal_en)
    {
    }
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
    if (conf.energy_cal_en)
    {
    }
    return torch::matmul(mat * conf.computeV, data.index({Slice(0, mat.size(2))}) * deltaConduct + conf.minConduct).div(rowSize * conf.maxConduct);
}

/**
 * Before you use mm. you need to call preWorkForMM first for pretreatment.
* mat is the data you want to perform mm. e.g. [3 0 4], but you should reverts it to voltage vector (might be multi pulses, so 1d tensor mat will return 2d tensor).
* @param mat input matrix. {batch_size, rowSize}
* @return Tensor. return 3d tensor {batch_size, inBits/inVBits, rowSize}.
*/
at::Tensor phyArrayPro::preWorkForMM(const at::Tensor &mat, pim_array_config *pconf, phy_array_config *conf, double &max_one)
{
    if (pconf->dynamic_max_input)
    {
        if (pconf->has_negative_input)
            max_one = std::min(pconf->max_phy_input_value, mat.abs().max().item<double>());
        else
            max_one = std::min(pconf->max_phy_input_value, mat.max().item<double>());
        // std::cout << "max_one = " << max_one << std::endl;
    }
    else
        max_one = pconf->max_phy_input_value;

    at::Tensor out;
    int inLevels = (1 << pconf->inBits);
    int inVLevels = (1 << pconf->inVBits);
    int nums = pconf->inBits / pconf->inVBits;
    double k = (inLevels - 1) / max_one;

    if (pconf->has_negative_input)
    {
        if (pconf->inBits <= 1)
        {
            throw("has neg input, inbits should be >1.");
        }
        auto index_larger = mat > max_one;
        auto index_smaller = mat < -max_one;
        int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
        int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();

        if (!pconf->trunc_input && num_larger_than_max + num_smaller_than_min != 0)
        {
            throw("trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array.");
        }
        if (pconf->inVBits == 1) // complement form to represent number
        {
            out = mat.add(max_one).mul(k / 2);
            out.index_put_({index_larger}, inLevels - 1);
            out.index_put_({index_smaller}, 0);

            out = out.add(0.5).to(torch::kInt32).bitwise_xor((1 << (pconf->inBits - 1)));
            max_one *= 2;
        }
        else //support negative voltage for negative number
        {
            auto neg = mat < 0;
            cout << neg << endl;
            out = mat.abs().mul(k).add(0.5).to(torch::kI32);
            out.index_put_({index_larger}, inLevels - 1);
            out.index_put_({index_smaller}, inLevels - 1);

            at::Tensor output = torch::empty({out.size(0), nums, out.size(1)}, TensorOptions(mat.device()).dtype(torch::kFloat64));
            at::Tensor neg_index = torch::empty({out.size(0), nums, out.size(1)}, TensorOptions(mat.device()).dtype(torch::kBool));
            int mask = inVLevels - 1;
            neg = neg.view({out.size(0), 1, out.size(1)});
            out = out.view({out.size(0), 1, out.size(1)});
            at::parallel_for(0, nums, 0, [&](int st, int ed) {
                for (int i = st; i < ed; ++i)
                {
                    output.index_put_({Slice(), Slice(i, i + 1)}, (out.__rshift__(i * pconf->inVBits).bitwise_and(mask).div((double)(inVLevels - 1))));
                    neg_index.index_put_({Slice(), Slice(i, i + 1)}, neg);
                }
            });
            
            return output.index_put({neg_index}, output.index({neg_index})*-1 );;
        }
    }
    else
    {
        auto index_larger = mat > max_one;
        auto index_smaller = mat < 0;
        int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
        int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();
        if (!pconf->trunc_input && num_larger_than_max + num_smaller_than_min != 0)
        {
            throw("trunc_input = false, && input exsits value mx than maximum or less than min. please modify config of array.");
        }
        out = mat.mul(k).add(0.5).to(torch::kInt32);
        out.index_put_({index_larger}, inLevels - 1);
        out.index_put_({index_smaller}, 0);
    }

    // out is {batch_size, rowSize}
    at::Tensor output = torch::empty({out.size(0), nums, out.size(1)}, TensorOptions(mat.device()).dtype(torch::kFloat64));
    int mask = inVLevels - 1;
    out = out.view({out.size(0), 1, out.size(1)});
    at::parallel_for(0, nums, 0, [&](int st, int ed) {
        for (int i = st; i < ed; ++i)
        {
            output.index_put_({Slice(), Slice(i, i + 1)}, (out.__rshift__(i * pconf->inVBits).bitwise_and(mask).div((double)(inVLevels - 1))) );
        }
    });

    return output;
}

/**
* @param mat: sizes {batch_size, inBits/inVbits, phyColSize}
* @param post: data will be used in post work
* @param max_one: data will be used in post work, max_one is send from prework
* @return tensor sizes {batch_size, unitNumPerPhyRow}
*/
at::Tensor phyArrayPro::postWorkForMM(const at::Tensor &mat, pim_array_config *pconf, phy_array_config *conf, double &max_one)
{
    int outLevels = 1 << pconf->outBits;
    int cellsPerUnit = pconf->unitBits / pconf->cellBits;

    at::Tensor out;

    if (pconf->mode == 1) // ref col mode
    {
        out = mat.mul(outLevels - 1).round(); //let out range from 0 -- outLevels -1
        out.index({Slice(), Slice(), Slice(0, out.size(3) - 1)}).subtract_(out.index({Slice(), Slice(), Slice(out.size(3) - 1)}));

        
    }
    else // postive & negative array mode,  in this single array, normal calculation.
    {
        /* code */
        int unitNumPerPhyRow = mat.size(3) / cellsPerUnit;
        int usedCellsPerRow = unitNumPerPhyRow * cellsPerUnit;
        at::Tensor output = torch::empty({mat.size(0), unitNumPerPhyRow});
        auto unitScalar = torch::ones({mat.size(1), usedCellsPerRow}, TensorOptions(mat.device()).dtype(torch::kF64));
        out = mat.index({Slice(), Slice(), Slice(0, usedCellsPerRow)}).mul(outLevels - 1).round(); //let out range from 0 -- outLevels -1

        at::parallel_for(0, cellsPerUnit, 0, [&](int st, int ed) -> void {
            for (int i = st; i < ed; ++i)
            {
                unitScalar.index({Slice(), Slice(i, usedCellsPerRow, cellsPerUnit)}).__ilshift__(i);
            }
        });

        at::parallel_for(0, unitScalar.size(1), 0, [&](int st, int ed) -> void {
            for (int i = st; i < ed; ++i)
            {
                unitScalar.index({Slice(i, i + 1)}).__ilshift__(i);
            }
        });
        out.mul_(unitScalar);
        out.sum(1);
    }
}

void phyArrayPro::print(std::ostream &os)
{
    using std::endl;
    os << "data = " << endl;
    os << data << endl;
    if (conf.energy_cal_en)
    {
        os << " total energy = " << readEnergy + writeEnergy
           << ", write energy = " << writeEnergy
           << ", read energy = " << readEnergy << endl;
    }
    os << " total write cnt = " << totalWrCnt << endl;
    if (conf.write_cnt_en)
    {
        os << " total cmp write cnt = " << totalCmpWrCnt << endl;
        os << "cell write cnt = " << endl;
        os << cellWrCnt << endl;
    }
}

#endif //PIMTORCH_PHYARRAYSIMPLE_H
