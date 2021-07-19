#ifndef PIMTORCH_IR_DROP_SOLVE_INTERFACE_H
#define PIMTORCH_IR_DROP_SOLVE_INTERFACE_H
#pragma once

#include <torch/torch.h>
#include <torch/custom_class.h>
#include "pim_array_config.h"
#include <vector>
#include <tuple>

using torch::indexing::Slice;

// this Ir_solver can only be used for computing
// all row should connected to a voltage (voltage 0 is ok which is the ground)
// all col should connected to ground
struct IR_solver
{
    int rowSize, colSize, n;
    double g_load, g_wire;
    IR_solver(const pim_array_pro_config *conf = &pro_decf())
    {
        this->conf = conf;
        if (conf->mode == 0)
            rowSize = conf->phyArrRowSize, colSize = conf->phyArrColSize;
        else if (conf->mode == 1)
            rowSize = conf->phyArrRowSize, colSize = conf->phyArrColSize+2;
        else
        {
            throw "mode not support";
        }
        if (conf->ir_drop.enable)
        {
            g_load = conf->ir_drop.g_load;
            g_wire = conf->ir_drop.g_wire;
        }
        else
        {
            g_load = 2.8;
            g_wire = 2.8;
        }
        
        init();
    }    

    IR_solver(int m, int n, float g_wire, float g_load): rowSize(m), colSize(n)\
        , g_wire(g_wire), g_load(g_load)
    {
        init();
    }
    void init();
    const pim_array_pro_config *conf;
    at::Tensor node, value;
    at::Tensor solve(const at::Tensor &inV, const at::Tensor &G_mat);
    at::Tensor solve_fast(const at::Tensor &inV, const at::Tensor &G_mat, int times);
};

void IR_solver::init()
{
    std::vector<int> x, y;
    std::vector<double> v;
    n = rowSize * colSize;
    x.reserve(n*2+rowSize*2+colSize*2+5);
    y.reserve(n*2+rowSize*2+colSize*2+5);
    v.reserve(n*2+rowSize*2+colSize*2+5);
    //build up every node
    //0~2*n should minus gij
    for (int i=0; i<n*2-colSize; ++i)
    {
        x.push_back(i);
        y.push_back(i);
        v.push_back(-2*g_wire);  //need to minus gij
    }
    for (int i=2*n-colSize; i<2*n; ++i)
    {
        x.push_back(i);
        y.push_back(i);
        v.push_back(-g_wire-g_load); //need to minus gij
    }
    for (int i=n*2; i<2*n+rowSize+colSize; ++i)
    {
        x.push_back(i);
        y.push_back(i);
        v.push_back(1);
    }
    for (int i=n*2+rowSize+colSize; i<2*n+rowSize*2+colSize*2; ++i)
    {
        x.push_back(i);
        y.push_back(i);
        v.push_back(-g_wire);
    }
    //set start = 2*n+rowSize*2+colSize*2
    // [start, start+2n) should set to gi,j 
    //cell G build up
    for (int i=0; i<n; ++i)
    {
        x.push_back(i);
        y.push_back(i+n);
        v.push_back(0);  //currently, i don't know the Gij, so set to 0
    }
    for (int i=0; i<n; ++i)
    {
        x.push_back(i+n);
        y.push_back(i);
        
        v.push_back(0);
    }

    //build up row wire conductance
    for (int i=0; i<rowSize; ++i)
    {
        for (int j=0; j<colSize-1; ++j)
        {
            int num=i*colSize+j;
            x.push_back(num);
            y.push_back(num+1);
            v.push_back(g_wire);

            x.push_back(num+1);
            y.push_back(num); 
            v.push_back(g_wire);
        }

        int num = i*colSize+colSize-1;
        int right = 2*n+rowSize+colSize+i;
        x.push_back(num);
        y.push_back(right);
        x.push_back(right);
        y.push_back(num);
        v.push_back(g_wire);
        v.push_back(g_wire);
    }

    //build up col wire conductance
    for (int i=0; i<rowSize-1; ++i)
    {
        for (int j=0; j<colSize; ++j)
        {
            int num = i*colSize+j+n;
            x.push_back(num);
            y.push_back(num+colSize);
            v.push_back(g_wire);

            x.push_back(num+colSize);
            y.push_back(num);
            v.push_back(g_wire);
        }
    }
    for (int j=0; j<colSize; ++j)
    {
        int num = j+n;
        int up = 2*n+rowSize*2+colSize+j;

        x.push_back(num);
        x.push_back(up);
        y.push_back(up);
        y.push_back(num);

        v.push_back(g_wire);
        v.push_back(g_wire);
    }

    //build up left wire conductance connected to first col
    for (int i=0; i<rowSize; ++i)
    {
        int num = i*colSize;
        int left = n*2+i;
        x.push_back(num);
        y.push_back(left);
        v.push_back(g_wire);
    }

    //build up down load (wire) conductance connected to last row
    for (int i=0; i<colSize; ++i)
    {
        int num = n+(rowSize-1)*colSize+i;
        int down = 2*n+rowSize+i;
        x.push_back(num);
        y.push_back(down);
        v.push_back(g_load);
    }

    
    node = torch::cat({torch::tensor(x), torch::tensor(y)}).view({2, (long)x.size()});
    value = torch::tensor(v);
}

at::Tensor IR_solver::solve(const at::Tensor &inV, const at::Tensor &G_mat)
{
    if (inV.sizes().size()==3)
    {
        int batch_size = inV.size(0);
        int row_size = inV.size(1);
        int col_size = inV.size(2);
        return solve(inV.reshape({-1, col_size}), G_mat).reshape({batch_size, row_size, -1});
    }
    at::Tensor B = torch::zeros({inV.size(0), 2*n+rowSize*2+colSize*2}, inV.options());
    B.index({Slice(), Slice(2*n, 2*n+rowSize)}) = inV;
    
    at::Tensor node = this->node.to(inV.device()).clone();
    at::Tensor value = this->value.to(inV).clone();
    at::Tensor g_mat_flat2 = torch::cat({G_mat.flatten(), G_mat.flatten()});
    int start = 2*(rowSize+colSize+n);
    value.index({Slice(0, 2*n)}) -= g_mat_flat2;
    value.index({Slice(start, start+2*n)}) = g_mat_flat2;
    //std::cout << node << std::endl;
    //std::cout << value << std::endl;
    at::Tensor A = torch::sparse_coo_tensor(node, value, {start, start}, inV.device());
    
    
    auto ans = torch::solve(B.t(), A.to_dense());
    //std::cout <<std::get<0>(ans).t() << std::endl;
    return (std::get<0>(ans)).t().index({Slice(), Slice(n*2-colSize, n*2)})*g_load;
}

//ir drop fast mode don't support negative voltage inputs
at::Tensor IR_solver::solve_fast(const at::Tensor &inV, const at::Tensor &G_mat, int times)
{
    if (inV.sizes().size()==3)
    {
        int batch_size = inV.size(0);
        int row_size = inV.size(1);
        int col_size = inV.size(2);
        return solve_fast(inV.reshape({-1, col_size}), G_mat, times).reshape({batch_size, row_size, -1});
    }

    int batch_size = inV.size(0);
    int row_size = inV.size(1);
    auto up_v = torch::empty({colSize, batch_size, row_size}, inV.options());
    auto down_v = torch::zeros({colSize, batch_size, row_size}, inV.options());
    auto g = G_mat.t().view({colSize, 1, row_size});
    up_v.index({Slice()}) = inV*0.7; 
    for (int i=0; i<times; ++i)
    {
        auto current = (up_v-down_v)*g;
        auto i_bitline = current.sum(0);
        auto i_wordline = current.sum(2, true);
        up_v[0] = inV - i_bitline/g_wire;

        i_bitline -= current[0];
        for (int c=1; c<colSize; ++c)
        {
            up_v[c]=up_v[c-1]-i_bitline/g_wire;
            i_bitline -= current[c];
        }

        up_v.index_put_({up_v<0}, 0.0);
        down_v.index({Slice(), Slice(), Slice(rowSize-1, rowSize)}) = i_wordline/g_load;
        i_wordline -= current.index({Slice(), Slice(), Slice(row_size-1)});
        for (int r=rowSize-2; r>=0; --r)
        {
            down_v.index({Slice(), Slice(), Slice(r, r+1)}) = down_v.index({Slice(), Slice(), Slice(r+1, r+2)})+i_wordline/g_wire;
            i_wordline -= current.index({Slice(), Slice(), Slice(r, r+1)});
        }
        auto d = down_v>up_v;
        down_v.index_put_({d}, up_v.index({d}));
        //std::cout << up_v <<std::endl;
        //std::cout << down_v <<std::endl;
        //std::cout << "-------------------" << std::endl;
    }
    auto current = (up_v-down_v)*g;
    return current.sum(2).t();
}


static IR_solver ir_solver;

/**
 * fast calculation
 * @param inV: input Voltage; 2d tensor {inPluses, V}
 * @param G_mat: cell conductance
 * @param rowSize: row size
 * @param colSize: col size
 * @param g_wire: wire resistance
 * @param g_load: adc load resistance
 * @return : output current
 */
at::Tensor ir_drop_solve_fast(const at::Tensor &inV, const at::Tensor &G_mat, int rowSize, int colSize, int times, double g_wire = -1, double g_load = -1)
{
    auto eq=[](double &x, double &y)->bool
    {
        return fabs(x-y)<1e-12;
    };
    if (g_wire<0 || g_load<0)
        return ir_solver.solve_fast(inV, G_mat, times);
    if (ir_solver.rowSize==rowSize && ir_solver.colSize==colSize && eq(g_wire, ir_solver.g_wire) && eq(g_load, ir_solver.g_load))    
        return ir_solver.solve_fast(inV, G_mat, times);
    else
    {
        IR_solver local_ir_solver(rowSize, colSize, g_wire, g_load);
        return local_ir_solver.solve_fast(inV, G_mat, times);
    }
}
/**
 * accuracy calculation, using linear solver.
 * @param inV: input Voltage; 2d tensor {inPluses, V}
 * @param G_mat: cell conductance
 * @param g_wire: wire conductance
 * @param g_load: adc load conductance 
 * @param rowSize: row_size
 * @param colSize: col_size
 * @return : output current
 */
at::Tensor ir_drop_solve_acc(const at::Tensor &inV, const at::Tensor &G_mat, int rowSize, int colSize, double g_wire = -1, double g_load = -1)
{
    auto eq=[](double &x, double &y)->bool
    {
        return fabs(x-y)<1e-12;
    };
    if (g_wire<0 || g_load<0)
        return ir_solver.solve(inV, G_mat);
    if (ir_solver.rowSize==rowSize && ir_solver.colSize==colSize && eq(g_wire, ir_solver.g_wire) && eq(g_load, ir_solver.g_load))    
        return ir_solver.solve(inV, G_mat);
    else
    {
        IR_solver local_ir_solver(rowSize, colSize, g_wire, g_load);
        return local_ir_solver.solve(inV, G_mat);
    }
}

/**
 * This Function check whether fast_solve is similar to solve, in the configuration of pim_array_pro.yaml.
 * fast_solve is not always efficent and may fail to calculate ir_drop problem (when ir_drop is quite big, fast_solve is more likely to be wrong) 
 * @param times: run times  
 * @param percent: loss should less than percent
 * @return: if loss<percent, true. Otherwises, false. 
 */
bool ir_drop_fastMode_check(int times, double percent)
{
    for (int i=0; i<times; ++i)
    {
        at::Tensor inV = torch::randn({1, 1, ir_solver.rowSize}, torch::kF64).abs()*ir_solver.conf->computeV;
        at::Tensor G = torch::ones({ir_solver.rowSize, ir_solver.colSize}, torch::kF64)*ir_solver.conf->maxConduct;
        auto ori = ir_solver.solve(inV, G);
        auto loss = (ir_solver.solve_fast(inV, G, ir_solver.conf->ir_drop.times)-ori)/ori;
        loss.abs();
        auto num = loss>(percent);
        if (num.sum(torch::kI32).item<int>()>0)
            return false;
    }
    for (int i=0; i<times; ++i)
    {
        at::Tensor inV = torch::randn({1, 1, ir_solver.rowSize}, torch::kF64).abs()*ir_solver.conf->computeV;
        at::Tensor G = torch::randn({ir_solver.rowSize, ir_solver.colSize}, torch::kF64).abs()*(ir_solver.conf->maxConduct-ir_solver.conf->minConduct)+ir_solver.conf->minConduct;
        auto ori = ir_solver.solve(inV, G);
        auto loss = (ir_solver.solve_fast(inV, G, ir_solver.conf->ir_drop.times)-ori)/ori;
        loss.abs();
        auto num = loss>(percent);
        if (num.sum(torch::kI32).item<int>()>0)
            return false;
    }
    return true;
}

#endif
