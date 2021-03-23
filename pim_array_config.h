#ifndef PIMTORCH_PIM_ARRAY_CONFIG_H
#define PIMTORCH_PIM_ARRAY_CONFIG_H

#include "yaml-cpp/yaml.h"

struct pim_array_config
{
    int32_t rowSize, colSize;                       //  logic array size
    int32_t phyArrRowSize, phyArrColSize;           //  phy array size, a logic array is formed by one or multiple phy arrays.
    int32_t inBits, outBits, unitBits, cellBits;    /*  input/output data bits.  unit bits means precision of data in array. cell bits means one memory cell's precision
                                                        e.g. unitBits = 8, cellBits = 2.  we need 4 memory cell to represent 1 unit.
                                                   */

    bool has_negative_input;                        // input value has negative number
    double max_phy_input_value;                     // input value has its maximum, we will use this maximum to regionalizatoin input value by inBits.
    bool trunc_input;                               // if true, the value > max_phy_input_value, will trunc to the max_phy_input_value. if false, if will reprot error if value>max_phy_input
    bool dynamic_max_input;                         // if true, we will dynamic get max_input rather than use max_phy_input_value
    
    pim_array_config(std::string filename = "../pimarray.yaml")
    {
        pim_array_config x;
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


const pim_array_config decf("../pimarray.yaml");

const pim_array_config decf_for_counters("../pim_onlyCounter.yaml");

#endif //PIMTORCH_PIM_ARRAY_CONFIG_H
