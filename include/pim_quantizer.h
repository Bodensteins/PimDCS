#ifndef PIMTORCH_PIM_QUANTIZER_H
#define PIMTORCH_PIM_QUANTIZER_H
#pragma once

#include <tuple>
#include <torch/torch.h>
#include <map>
#include "pim_array_config.h"
#include <torch/custom_class.h>

namespace PIM {

using torch::TensorOptions;
using namespace torch::indexing;
using std::tuple;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

template<class T>
struct baseQ
{
    virtual T quan(at::Tensor) = 0;
    virtual at::Tensor dequan(T) = 0;
};

/**
 * Pos & Neg array mode for storing matrix.
 * 
 * 
 * */
struct arrPNQuantizer: baseQ<pair<at::Tensor, at::Tensor>>
{
    const double w_max;   // read only after initial
    const int levels;     // read-only after initial

    arrPNQuantizer(const pim_array_pro_config *conf = &pro_decf()): w_max(conf->max_weight_value), levels(conf->unitLevels) {}

    pair<at::Tensor, at::Tensor> quan(at::Tensor x)
    {
        double mx = x.abs().max().item<double>();
        auto y = x.div(w_max);
        if (mx>w_max)
        {
            y.index_put_({y > 1}, 1.0);
            y.index_put_({y < -1}, -1.0);
        }

        auto pos = y.mul(levels-1);
        auto neg = pos.clone();

        pos.index_put_({pos < 0}, 0); // 0 ~ unitLevels-1
        neg.index_put_({neg > 0}, 0);
        neg.abs_(); // 0 ~ unitLevels-1
    
        pos = pos.add(0.5).to(torch::kInt32);
        neg = neg.add(0.5).to(torch::kInt32);

        return std::make_pair(pos, neg);
    }

    at::Tensor dequan(pair<at::Tensor, at::Tensor> x)
    {
        return (x.first-x.second).mul(w_max/(levels-1));
    }
};

/**
 * Ref column mode for storing matrix.
 * 
 * */
struct arrRefQuantizer: baseQ<at::Tensor>
{
    const double w_max;   //read-only after initial
    const int levels;     //read-only after initial

    arrRefQuantizer(const pim_array_pro_config *conf = &pro_decf()): w_max(conf->max_weight_value), levels(conf->unitLevels) {}

    at::Tensor quan(at::Tensor x)
    {
        double mx = x.abs().max().item<double>();
        auto y = x.div(w_max);
        if (mx>w_max)
        {
            y.index_put_({y > 1}, 1.0);
            y.index_put_({y < -1}, -1.0);
        }
        return y.add(1).mul((levels - 1)/2.0).add(0.5).to(torch::kInt32);
    }

    at::Tensor dequan(at::Tensor x)
    {
        return x.div((levels - 1) / 2.0).subtract(1.0).mul(w_max);
    }
};


struct ioQuantizer: baseQ<pair<at::Tensor, double>>
{
    const pim_array_pro_config *conf;
    ioQuantizer(const pim_array_pro_config *conf = &pro_decf())
    {
        this->conf = conf;
    }

    pair<at::Tensor, double> quan(at::Tensor);
    at::Tensor dequan(pair<at::Tensor, double>);
};




/**
 * Before you use mm. you need to call preWorkForMM first for pretreatment.
* mat is the data you want to perform mm. e.g. [3 0 4], but you should reverts it to voltage vector (might be multi pulses, so 1d tensor mat will return 2d tensor).
* @param x input matrix. {batch_size, rowSize}
* @return Tensor. return 3d tensor {batch_size, inBits/inVBits, rowSize}.
*/
pair<at::Tensor, double> ioQuantizer::quan(at::Tensor x)
{
    double x_max = x.abs().max().item<double>();
    double mx;
    if (conf->dynamic_max_input)
        mx = std::min(mx, x_max);
    else
        mx = conf->max_phy_input_value;

    at::Tensor out;

    // int nums = conf->inBits / conf->inVBits;
    double k = (conf->inLevels - 1) / mx;

    if (conf->has_negative_input)
    {
        if (conf->inBits <= 1)
        {
            cout << "has neg input, inbits should be >1." << endl;
            throw("has neg input, inbits should be >1.");
        }

        at::Tensor index_larger;
        at::Tensor index_smaller;
        bool ok = false;
        if (!conf->trunc_input)
        {
            ok = true;
            index_larger = x > mx;
            index_smaller = x < -mx;
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
            out = x.add(mx).mul(k / 2);
            if (x_max>mx)
            {
                if (!ok)
                {
                    index_larger = x > mx;
                    index_smaller = x < -mx;
                }
                out.index_put_({index_larger}, conf->inLevels - 1);
                out.index_put_({index_smaller}, 0);
            }
            out = out.add(0.5).to(torch::kInt32).bitwise_xor((1 << (conf->inBits - 1)));
            mx *= 2;
        }
        else //support negative voltage for negative number
        {
            auto neg = x < 0;
            
            out = x.abs().mul(k).add(0.5).to(torch::kI32);
            if (x_max>mx)
            {
                auto index_larger = x > mx;
                auto index_smaller = x < -mx;
                out.index_put_({index_larger}, conf->inLevels - 1);
                out.index_put_({index_smaller}, conf->inLevels - 1);
            }
            at::Tensor output = torch::empty({out.size(0), conf->inPluses, out.size(1)}, TensorOptions(x.device()).dtype(torch::kFloat64));
            int mask = conf->inVLevels - 1;
            neg = neg.view({out.size(0), 1, out.size(1)});
            auto neg_index = torch::cat({std::vector<at::Tensor>(conf->inPluses, neg)}, 1);

            out = out.view({out.size(0), out.size(1), 1});
            out = torch::cat({std::vector<at::Tensor>(conf->inPluses, out)}, 2);
            output.index({Ellipsis}) = out.__irshift__(conf->rshift.to(x.device())).bitwise_and_(mask).transpose_(1, 2);
  
            return make_pair(output.index_put({neg_index}, output.index({neg_index})*-1 ), mx);
        }
    }
    else // input vector should be positive
    {

        if (!conf->trunc_input)
        {
            auto index_larger = x > mx;
            auto index_smaller = x < 0;
            int num_larger_than_max = index_larger.sum(torch::kInt32).item<int>();
            int num_smaller_than_min = index_smaller.sum(torch::kInt32).item<int>();
            //check if there exists values that are larger (or smaller) than max_one (or 0).
            if (num_larger_than_max + num_smaller_than_min != 0)
            {
                cout << "trunc_input = false, && input exsits value larger than maximum or less than min. please modify config of array." << endl;
                throw("trunc_input = false, && input exsits value larger than maximum or less than min. please modify config of array.");
            }
            out = x.mul(k).add(0.5).to(torch::kInt32);
        }
        else
        {
            out = x.mul(k).add(0.5).to(torch::kInt32);
            if (x_max>mx)
            {
                auto index_larger = x > mx;
                auto index_smaller = x < 0;
                out.index_put_({index_larger}, conf->inLevels - 1);
                out.index_put_({index_smaller}, 0);
            }
        }
    }

    // out is {batch_size, rowSize}
    at::Tensor output = torch::empty({out.size(0), conf->inPluses, out.size(1)}, TensorOptions(x.device()).dtype(torch::kFloat64));
    int mask = conf->inVLevels - 1;
    out = out.view({out.size(0), out.size(1), 1});
    out = torch::cat({std::vector<at::Tensor>(conf->inPluses, out)}, 2);
    output.index({Ellipsis}) = out.__irshift__(conf->rshift.to(x.device())).bitwise_and_(mask).transpose_(1, 2);

    return make_pair(output, mx);
}

/**
* @param input: first value is mat: sizes {batch_size, inBits/inVbits, phyColSize}
* @param input: second value is max_one: data will be used in post work, max_one is send from prework
* @return tensor sizes {batch_size, unitNumPerPhyRow}
*/
at::Tensor ioQuantizer::dequan(pair<at::Tensor, double> input)
{ 
    auto mat = input.first;
    double max_one = input.second;
    double scalar_value = max_one / (conf->unitLevels-1) * conf->max_weight_value / (conf->inLevels-1);
    at::Tensor out;

    if (conf->mode == 1) // ref col mode
    {
        if (conf->maxCurrentNum>=conf->outLevels)  
        {
            out = mat.div(conf->deltaConduct*conf->adc_scalar).round_().mul_(conf->adc_scalar); // 0---maxCurrentNum
        }
        else
            out = mat.div(conf->deltaConduct).round_(); //let out range from 0 -- outLevels -1, double

        out.index({Ellipsis, Slice(0, conf->phyArrColSize+1)}).subtract_(out.index({Ellipsis, Slice(conf->phyArrColSize+1)}));
        
        at::Tensor refValue = out.index({Ellipsis, Slice(conf->phyArrColSize, conf->phyArrColSize+1 ) });

        refValue = refValue.__lshift__(conf->unitBits).subtract_(refValue);

        out = out.index({Slice(), Slice(), Slice(0, conf->usedCellsPerPhyRow)});

        at::Tensor output = torch::empty({mat.size(0), conf->inPluses, conf->unitsPerPhyRow}, mat.device());

        
        out *= conf->unitScalar.to(mat.device());
        
        output.index({Ellipsis}) = out.transpose_(2, 1).view({mat.size(0), conf->unitsPerPhyRow, conf->cellsPerUnit, conf->inPluses}).sum(2).transpose_(1, 2);

        output = output.__lshift__(1)-refValue;

        out = (output* conf->inScalar.to(mat.device())).sum(1).mul_(scalar_value);  //{batch_size, unitNumPerPhyRow}
        return out;
    }
    else // postive & negative array mode,  in this single array, normal calculation.
    {

        if (conf->maxCurrentNum>=conf->outLevels)
        {
            out = mat.div(conf->deltaConduct*conf->adc_scalar).round_().mul_(conf->adc_scalar).index({Ellipsis, Slice(0, conf->usedCellsPerPhyRow)}); // 0---maxCurrentNum
        }
        else
            out = mat.div(conf->deltaConduct).round_().index({Ellipsis, Slice(0, conf->usedCellsPerPhyRow)}); //let out range from 0 -- maxCurrentNum, double
        at::Tensor output = torch::empty({mat.size(0), conf->inPluses, conf->unitsPerPhyRow}, mat.device());

        out *= conf->unitScalar.to(mat.device());
        
        output.index({Ellipsis}) = out.transpose_(2, 1).view({mat.size(0), conf->unitsPerPhyRow, conf->cellsPerUnit, conf->inPluses}).sum(2).transpose_(1, 2);

        out = (output * conf->inScalar.to(mat.device())).sum(1).mul_(scalar_value); //{batch_size, unitNumPerPhyRow}
        return out;
    }
}

}

#endif