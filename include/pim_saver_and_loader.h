//
// Created by 周恒 on 2021/7/26.
//

#ifndef PIMTORCH_PIM_SAVER_AND_LOADER_H
#define PIMTORCH_PIM_SAVER_AND_LOADER_H
#pragma once

#include "pim_linear.h"
#include "pim_conv.h"

namespace PIM
{
    void pim_saver(torch::nn::Module &model, const std::string& file_path) {
        torch::save(model.parameters(), file_path);
        std::cout << "Save model parameters to " << file_path << std::endl;
    }

    /**
     * load parameters from pre-trained model.
     * @param model
     * @param param_vec
     */
    void pim_loader(torch::nn::Module &model, const std::string& file_path) {
        std::vector<torch::Tensor> params_vec;
        torch::load(params_vec, file_path);
        std::vector<torch::Tensor> model_param_vec = model.parameters();
        for (int i = 0; i < model_param_vec.size(); i++) {
            model_param_vec[i].set_data(params_vec[i]);
        }
        model.apply([](torch::nn::Module& module) {
            if (PIM::PimLinearImpl* module_ptr = dynamic_cast<PIM::PimLinearImpl*>(&module)) {
                module_ptr->sync_weight();
            } else if (PIM::PimConv2dImpl* module_ptr = dynamic_cast<PIM::PimConv2dImpl*>(&module)) {
                module_ptr->sync_weight();
            }
        });
        std::cout << "Load and sync model patameter from " << file_path << std::endl;
    }
}
#endif //PIMTORCH_PIM_SAVER_AND_LOADER_H
