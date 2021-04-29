#ifndef PIMTORCH_PIM_ARRAY_CONFIG_H
#define PIMTORCH_PIM_ARRAY_CONFIG_H

#include "yaml-cpp/yaml.h"
#include <torch/torch.h>
#include <torch/custom_class.h>
#include <cmath>
using torch::indexing::Slice;

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
    
    pim_array_config(std::string filename = "../config/pimarray.yaml")
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
    }

    void print(std::ostream & os)
    {
        os << "phyArr sizes = [" << phyArrRowSize << ',' << phyArrColSize << "]\n" \
            << "in/out/unit bits" "= [" << inBits << '/' << outBits << '/' << unitBits << "]\n" \
            << "has neg " << has_negative_input << ',' << "max_phy_input =" << max_phy_input_value \
            << "trunc_input " << trunc_input << "dynamic_max " << dynamic_max_input << std::endl;
    }
};


const pim_array_config decf("../config/pimarray.yaml");

const pim_array_config decf_for_counters("../config/pim_onlyCounter.yaml");

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




struct pim_array_pro_config
{
    phy_array_readMode rm;
    phy_array_writeMode wm;
    double writeV, readV, computeV;
    bool C2C_en, D2D_en, nonLinearIV_en, write_cnt_en;/*, energy_cal_en;*/
    double C2C_theta;
    double minConduct, maxConduct;
    int32_t phyArrRowSize, phyArrColSize;           //  phy array size, a logic array is formed by one or multiple phy arrays.
    int32_t inBits, inVBits, outBits, unitBits, cellBits;    /*  input/output data bits.  unit bits means precision of data in array. cell bits means one memory cell's precision
                                                        e.g. unitBits = 8, cellBits = 2.  we need 4 memory cell to represent 1 unit.
                                                   */

    bool has_negative_input;                        // input value has negative number
    double max_phy_input_value;                     // input value has its maximum, we will use this maximum to regionalizatoin input value by inBits.
    double max_weight_value;                        // weight value has its maximum, we will use this maximum to regionalizatoin weight value by unitBits.
    bool trunc_input;                               // if true, the value > max_phy_input_value, will trunc to the max_phy_input_value. if false, if will reprot error if value>max_phy_input
    bool dynamic_max_input;                         // if true, we will dynamic get max_input rather than use max_phy_input_value

    int mode;   
    int cellsPerUnit, unitsPerPhyRow, usedCellsPerPhyRow;
    int inLevels, inVLevels, outLevels, unitLevels, cellLevels, inPluses;

    struct latency_params
    {
        bool enable;
        int parMMPhyNum;
        int parWrPhyNum;
        int parPhyWrSize;
        double phyWrLatency;
        double phyMMLatency;
        double phyReLatency;
        double addLatency;
        int addTreeWideSize;
        double latencyWrSinglePhyArr;
        double addTreeLatency;
        int addTreeSharedNum;
    }latency;

    struct energy_params
    {
        bool enable;
    }energy;

    at::Tensor inScalar, unitScalar;

    /* area config */
    double F;
    double share_peripheral;
    double array_peripheral;
    double DAC_area;
    double ADC_area;
    int share_inf_row_size;
    int share_inf_col_size;
    int row_share_subarray;
    int col_share_subarray;

    pim_array_pro_config(std::string filename = "../config/pim_array_pro.yaml")
    {
        YAML::Node config = YAML::LoadFile(filename);

        rm = static_cast<phy_array_readMode>(config["phy_array_readMode"].as<int>());
        wm = static_cast<phy_array_writeMode>(config["phy_array_writeMode"].as<int>());

        writeV = config["writeV"].as<double>();
        readV = config["readV"].as<double>();
        computeV = config["computeV"].as<double>();

        C2C_en = config["C2C_en"]["enable"].as<bool>();
        if (C2C_en)
            C2C_theta = config["C2C_en"]["theta"].as<double>(); 
        D2D_en = config["D2D_en"].as<bool>();
        nonLinearIV_en = config["nonLinearIV_en"].as<bool>();
        write_cnt_en = config["write_cnt_en"].as<bool>();
        //energy_cal_en = config["energy_cal_en"].as<bool>();

        cellBits = config["cellBits"].as<int>();
        minConduct = config["minConduct"].as<double>();
        maxConduct = config["maxConduct"].as<double>();


        phyArrRowSize = config["phyArrRowSize"].as<int>();
        phyArrColSize = config["phyArrColSize"].as<int>();
        inBits = config["inBits"].as<int>();
        outBits = config["outBits"].as<int>();
        unitBits = config["unitBits"].as<int>();
        has_negative_input = config["has_negative_input"].as<bool>();
        max_phy_input_value = config["max_phy_input_value"].as<double>();
        max_weight_value = config["max_weight_value"].as<double>();
        trunc_input = config["trunc_input"].as<bool>();
        dynamic_max_input = config["dynamic_max_input"].as<bool>();
        mode = config["mode"].as<int>();
        inVBits = config["inVBits"].as<int>();

        cellsPerUnit = unitBits / cellBits;
        unitsPerPhyRow = phyArrColSize / cellsPerUnit;
        usedCellsPerPhyRow = unitsPerPhyRow * cellsPerUnit;
        inLevels = 1 << inBits;
        inVLevels = 1 << inVBits;
        outLevels = 1 << outBits;
        unitLevels = 1 << unitBits;
        cellLevels = 1 << cellBits;
        // to calculate energy, we need write cnt
//        if (energy_cal_en)
//            write_cnt_en = true;

        inPluses = inBits/inVBits;

        unitScalar = torch::ones({inPluses, usedCellsPerPhyRow}, torch::kI32);

        at::parallel_for(0, cellsPerUnit, 0, [&](int st, int ed) -> void {
            for (int i = st; i < ed; ++i)
            {
                unitScalar.index({Slice(), Slice(i, usedCellsPerPhyRow, cellsPerUnit)}).__ilshift__(i*cellBits);
            }
        });

        inScalar = torch::ones({inPluses, 1}, torch::kF64);
        
        at::parallel_for(0, inPluses, 0, [&](int st, int ed) -> void {
            for (int i = st; i < ed; ++i)
            {
                inScalar.index_put_({Slice(i, i + 1)}, 1 << (i*inVBits));
            }
        });
        if (inVBits==1 && has_negative_input)
            inScalar.index_put_({Slice(inBits-1)}, (1 << (inBits-1))*-1);
        

        //-----latency params setting----
        latency.enable = config["latency_cal"]["enable"].as<bool>();
        if (latency.enable)
        {
            latency.parMMPhyNum = config["latency_cal"]["parMMPhyNum"].as<int>();
            latency.parWrPhyNum = config["latency_cal"]["parWrPhyNum"].as<int>();

            latency.parPhyWrSize = config["latency_cal"]["parPhyWrSize"].as<int>();
            
            latency.phyWrLatency = config["latency_cal"]["phyWrLatency"].as<double>();
            latency.phyMMLatency = config["latency_cal"]["phyMMLatency"].as<double>();
            latency.phyReLatency = config["latency_cal"]["phyReLatency"].as<double>();
            
            latency.addLatency = config["latency_cal"]["addLatency"].as<double>();
            latency.addTreeWideSize  = config["latency_cal"]["addTreeWideSize"].as<int>();
            
            if (latency.parPhyWrSize <=0 || latency.parPhyWrSize>phyArrRowSize)
                latency.parPhyWrSize = phyArrRowSize;
            
            latency.latencyWrSinglePhyArr = phyArrRowSize*phyArrColSize/latency.parPhyWrSize * latency.phyWrLatency;
            latency.addTreeLatency = latency.addLatency*std::log2(1.0*latency.addTreeWideSize);
            latency.addTreeSharedNum = config["latency_cal"]["addTreeSharedNum"].as<int>();
        }

        //energy params setting
        energy.enable = latency.enable && config["energy_cal"]["enable"].as<bool>();//must support latency
        if (energy.enable)
        {
            //todo:
        }

        // area config
        F = config["area"]["F"].as<double>();
        share_peripheral = config["area"]["share_peripheral"].as<double>();
        array_peripheral = config["area"]["array_peripheral"].as<double>();
        DAC_area = config["area"]["DAC_area"].as<double>();
        ADC_area = config["area"]["ADC_area"].as<double>();
        share_inf_row_size = config["area"]["share_inf_row_size"].as<int>();
        share_inf_col_size = config["area"]["share_inf_col_size"].as<int>();
        row_share_subarray = config["area"]["row_share_subarray"].as<int>();
        col_share_subarray = config["area"]["col_share_subarray"].as<int>();
    }
};

const pim_array_pro_config pro_decf("../config/pim_array_pro.yaml");

#endif //PIMTORCH_PIM_ARRAY_CONFIG_H
