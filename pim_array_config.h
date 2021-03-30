#ifndef PIMTORCH_PIM_ARRAY_CONFIG_H
#define PIMTORCH_PIM_ARRAY_CONFIG_H

#include "yaml-cpp/yaml.h"

struct pim_array_config
{
    int32_t rowSize, colSize;                       //  logic array size
    int32_t phyArrRowSize, phyArrColSize;           //  phy array size, a logic array is formed by one or multiple phy arrays.
    int32_t inBits, inVBits, outBits, unitBits, cellBits;    /*  input/output data bits.  unit bits means precision of data in array. cell bits means one memory cell's precision
                                                        e.g. unitBits = 8, cellBits = 2.  we need 4 memory cell to represent 1 unit.
                                                   */

    bool has_negative_input;                        // input value has negative number
    double max_phy_input_value;                     // input value has its maximum, we will use this maximum to regionalizatoin input value by inBits.
    double max_weight_value;                        // weight value has its maximum, we will use this maximum to regionalizatoin weight value by unitBits.
    bool trunc_input;                               // if true, the value > max_phy_input_value, will trunc to the max_phy_input_value. if false, if will reprot error if value>max_phy_input
    bool dynamic_max_input;                         // if true, we will dynamic get max_input rather than use max_phy_input_value

    int mode;                                       // 0-> p/n array, 1 -> ref col
    
    pim_array_config(std::string filename = "../pimarray.yaml")
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
        max_weight_value = config["max_weight_value"].as<double>();
        trunc_input = config["trunc_input"].as<bool>();
        dynamic_max_input = config["dynamic_max_input"].as<bool>();
        mode = config["mode"].as<int>();
        inVBits = config["inVBits"].as<int>();
    }

    void print(std::ostream & os)
    {
        os << "phyArr sizes = [" << phyArrRowSize << ',' << phyArrColSize << "]\n" \
            << "in/out/unit bits" "= [" << inBits << '/' << outBits << '/' << unitBits << "]\n" \
            << "has neg " << has_negative_input << ',' << "max_phy_input =" << max_phy_input_value \
            << "trunc_input " << trunc_input << "dynamic_max " << dynamic_max_input << std::endl;
    }
};


const pim_array_config decf("../pimarray.yaml");

const pim_array_config decf_for_counters("../pim_onlyCounter.yaml");

enum struct phy_array_writeMode
{
    V_Div_2 = 0,
    V_Div_3,
    Pluse
};

enum struct phy_array_readMode
{
    V_Div_2 = 0,
    V_Div_3,
    Ground
};

struct phy_array_config
{
    phy_array_readMode rm;
    phy_array_writeMode wm;
    double writeV, readV, computeV;
    bool C2C_en, D2D_en, nonLinearIV_en, write_cnt_en, energy_cal_en;
    int cellBits;
    double minConduct, maxConduct;
    phy_array_config(std::string filename = "../phy_array.yaml")
    {
        YAML::Node config = YAML::LoadFile(filename);

        rm = static_cast<phy_array_readMode>(config["phy_array_readMode"].as<int>());
        wm = static_cast<phy_array_writeMode>(config["phy_array_writeMode"].as<int>());

        writeV = config["writeV"].as<double>();
        readV = config["readV"].as<double>();
        computeV = config["computeV"].as<double>();

        C2C_en = config["C2C_en"].as<bool>();
        D2D_en = config["D2D_en"].as<bool>();
        nonLinearIV_en = config["nonLinearIV_en"].as<bool>();
        write_cnt_en = config["write_cnt_en"].as<bool>();
        energy_cal_en = config["energy_cal_en"].as<bool>();

        cellBits = config["cellBits"].as<int>();
        minConduct = config["minConduct"].as<double>();
        maxConduct = config["maxConduct"].as<double>();
    }
};

const phy_array_config phy_decf("../phy_array.yaml");

#endif //PIMTORCH_PIM_ARRAY_CONFIG_H
