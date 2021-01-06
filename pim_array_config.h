//
// Created by chenghuan on 1/6/21.
//

#ifndef PIMTORCH_PIM_ARRAY_CONFIG_H
#define PIMTORCH_PIM_ARRAY_CONFIG_H

#include "yaml-cpp/yaml.h"

struct pim_array_config
{
    int32_t rowSize, colSize;
    int32_t phyArrRowSize, phyArrColSize;
    int32_t inBits, outBits, unitBits, cellBits;

    bool has_negative_input;
    double max_phy_input_value;
    bool trunc_input;
    bool dynamic_max_input;

    pim_array_config(const std::string filename = "../pimarray.yaml")
    {
        YAML::Node config = YAML::LoadFile(filename);
        rowSize = config["rowSize"].as<int>();
        colSize = config["colSize"].as<int>();
        phyArrRowSize = config["phyArrRowSize"].as<int>();
        phyArrColSize = config["phyArrColSize"].as<int>();
        inBits = config["inBits"].as<int>();
        outBits = config["outBits"].as<int>();
        unitBits = config["unitBits"].as<int>();
        cellBits = config["cellBits"].as<int>();
        has_negative_input = config["has_negative_input"].as<bool>();
        max_phy_input_value = config["max_phy_input_value"].as<double>();
        trunc_input = config["trunc_input"].as<bool>();
        dynamic_max_input = config["dynamic_max_input"].as<bool>();
    }
};

extern const pim_array_config global_array_config;//global config


#endif //PIMTORCH_PIM_ARRAY_CONFIG_H
