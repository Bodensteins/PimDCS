#ifndef PIMTORCH_PIM_ARRAY_CONFIG_H
#define PIMTORCH_PIM_ARRAY_CONFIG_H

#include "config_path.h"
#include "yaml-cpp/yaml.h"
#include <torch/torch.h>
#include <torch/custom_class.h>
#include <cmath>
using torch::indexing::Slice;


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

    struct latency_area_params
    {
        int phyArrayNum;

        //1 means every phy array has a set of dac/adc.  2 means, 2 phy array share 1 set of dac/adc and so on.
        //1 set of dac/adc means phyArray #rowsize DACs, and phyArray #colSize ADCs.
        int adc_shared_ratio;
        int dac_shared_ratio;

        double dac_latency, adc_latency;
        double phyMMLatency;
        double phyRdLatency;
        double phyWrLatency;
        int parWrPhyNum;
        int parPhyWrSize;

        double single_SH_area, SH_area;
        // area unit is um^2
        double single_dac_area, single_adc_area, total_dac_area, total_adc_area;
        //total_dac_adc_area = (single_dac_area*phyArray_row_size + singe_adc_area*phyArray_col_size)*phyArrayNum/addaSharedRatio.
        double adder_area;
        double adder_latency;                  //no master how deep the tree is , please give the total latency here.
        double cell_area, single_phyArray_area, total_phyArray_area; //phyArrayNum * single_phy_array_area

        double PE_area;               //total_phyArray_area + adder_tree_area + total_dac_adc_area
        double latencyWrSinglePhyArr; // = phyArrRowSize*phyArrColSize/latency.parPhyWrSize * latency.phyWrLatency;
    }lat_area;

    struct energy_params
    {
        bool enable;
        double readRowPeripheryEnergy;
        double readColPeripheryEnergy;
        double writeRowPeripheryEnergy;
        double writeColPeripheryEnergy;
        double DACEnergy;
        double ADCEnergy;
        double computeRowPeripheryEnergy;
        double computeColPeripheryEnergy;
        double adderEnergy;
        bool readUseProbability;
        bool writeUseProbability;
        bool computeUseProbability;
        std::vector<double> CellPD;
        int CellPDDefault;
        std::vector<double> inVPD;
        int inVPDDefault;
        int writeParallelism;
        std::vector<double> averageEnergyPerWrite;
        //double averageEnergyPerWrite;
    }energy;

    struct SA_params
    {
        bool enable, runtime_en;
        double init_pSA0, init_pSA1;
        double pSA0, pSA1;
    }sa;

    struct ir_drop_params
    {
        bool enable, fast_mode;
        int times;
        double g_load, g_wire;
    }ir_drop;
    at::Tensor inScalar, unitScalar, rshift, crshift;

    /* area config */
    double cell_area;
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
        
        rshift = torch::ones({inPluses}, torch::kI32);
        for (int i=0; i<inPluses; ++i)
            rshift[i] = i*inVBits;
        
        crshift = torch::empty({cellsPerUnit}, torch::kI32);
        for (int i=0; i<cellsPerUnit; ++i)
            crshift[i] = i*cellBits;
        //-----latency params setting----
        {
            lat_area.phyArrayNum = config["latency_area_cal"]["phyArrayNum"].as<int>();
            lat_area.dac_shared_ratio = config["latency_area_cal"]["dac_shared_ratio"].as<int>();
            lat_area.adc_shared_ratio = config["latency_area_cal"]["adc_shared_ratio"].as<int>();
            
            if (lat_area.phyArrayNum%lat_area.dac_shared_ratio!=0 \
                ||lat_area.phyArrayNum%lat_area.adc_shared_ratio!=0)
            {
                std::cerr << "phyArrayNum should be divided by addaSharedRatio" << std::endl;
                exit(-1);
            }
            lat_area.adc_latency = config["latency_area_cal"]["adc_latency"].as<double>();
            lat_area.dac_latency = config["latency_area_cal"]["dac_latency"].as<double>();

            lat_area.phyMMLatency = config["latency_area_cal"]["phyMMLatency"].as<double>();
            lat_area.phyRdLatency = config["latency_area_cal"]["phyRdLatency"].as<double>();
            lat_area.phyWrLatency = config["latency_area_cal"]["phyWrLatency"].as<double>();
            lat_area.parWrPhyNum = config["latency_area_cal"]["parWrPhyNum"].as<int>();
            lat_area.parPhyWrSize = config["latency_area_cal"]["parPhyWrSize"].as<int>();
            lat_area.single_dac_area = config["latency_area_cal"]["single_dac_area"].as<double>();
            lat_area.single_SH_area= config["latency_area_cal"]["single_SH_area"].as<double>();
            lat_area.single_adc_area = config["latency_area_cal"]["single_adc_area"].as<double>();
            lat_area.adder_area = config["latency_area_cal"]["adder_area"].as<double>();
            lat_area.adder_latency = config["latency_area_cal"]["adder_latency"].as<double>();
            lat_area.cell_area = config["latency_area_cal"]["cell_area"].as<double>();

            lat_area.total_dac_area = (lat_area.single_dac_area*phyArrRowSize)*lat_area.phyArrayNum/lat_area.dac_shared_ratio;
            lat_area.total_adc_area = (lat_area.single_adc_area*phyArrColSize)*lat_area.phyArrayNum/lat_area.adc_shared_ratio;
            
            lat_area.single_phyArray_area = lat_area.cell_area * phyArrRowSize*phyArrColSize;
            lat_area.total_phyArray_area = lat_area.single_phyArray_area * lat_area.phyArrayNum;
            lat_area.SH_area = lat_area.single_SH_area*lat_area.phyArrayNum*phyArrColSize;

            lat_area.PE_area = lat_area.total_phyArray_area + lat_area.adder_area + lat_area.total_dac_area + lat_area.total_adc_area + lat_area.SH_area;

            lat_area.latencyWrSinglePhyArr = phyArrRowSize*phyArrColSize/lat_area.parPhyWrSize*lat_area.phyWrLatency;

        }

        //energy params setting
        energy.enable = config["energy_cal"]["enable"].as<bool>();//must support latency
        if (energy.enable)
        {
            //todo:may modify or add sth
            energy.readRowPeripheryEnergy = config["energy_cal"]["readRowPeripheryEnergy"].as<double>();
            energy.readColPeripheryEnergy = config["energy_cal"]["readColPeripheryEnergy"].as<double>();
            energy.writeRowPeripheryEnergy = config["energy_cal"]["writeRowPeripheryEnergy"].as<double>();
            energy.writeColPeripheryEnergy = config["energy_cal"]["writeColPeripheryEnergy"].as<double>();
            energy.computeRowPeripheryEnergy = config["energy_cal"]["computeRowPeripheryEnergy"].as<double>();
            energy.computeColPeripheryEnergy = config["energy_cal"]["computeColPeripheryEnergy"].as<double>();
            energy.DACEnergy = config["energy_cal"]["DACEnergy"].as<double>();
            energy.ADCEnergy = config["energy_cal"]["ADCEnergy"].as<double>();
            energy.computeRowPeripheryEnergy = config["energy_cal"]["computeRowPeripheryEnergy"].as<double>();
            energy.computeColPeripheryEnergy = config["energy_cal"]["computeColPeripheryEnergy"].as<double>();
            energy.readUseProbability = config["energy_cal"]["readUseProbability"].as<bool>();
            energy.writeUseProbability = config["energy_cal"]["writeUseProbability"].as<bool>();
            energy.computeUseProbability = config["energy_cal"]["computeUseProbability"].as<bool>();
            energy.writeParallelism = config["energy_cal"]["writeParallelism"].as<int>();

            if (energy.readUseProbability || energy.writeUseProbability || energy.computeUseProbability)
            {
                if (config["energy_cal"]["CellPD"])
                {
                    energy.CellPD = config["energy_cal"]["CellPD"].as<std::vector<double>>();
                    double sum = 0;
                    for (auto x:energy.CellPD)
                    {
                        sum += x;
                    }

                    double epsilon = 1e-5;
                    if (energy.CellPD.size() != cellLevels)
                    {
                        std::cerr << "CellPD illegal: the number of probabilities is not cellLevels" << std::endl;
                        exit(-1);
                    }
                    else if (fabs(sum - 1.0) > epsilon)
                    {
                        std::cerr << "CellPD illegal: the sum of probabilities is not 1" << std::endl;
                        exit(-1);
                    }
                }
                else
                {
                    energy.CellPDDefault = config["energy_cal"]["CellPDDefault"].as<int>();
                    if (energy.CellPDDefault == 0)
                    {
                        energy.CellPD = std::vector<double>(cellLevels, 1.0/cellLevels);
                    }
                    else if (energy.CellPDDefault == 1)
                    {
                        energy.CellPD = std::vector<double>(cellLevels, 0);
                        energy.CellPD[cellLevels - 1] = 1.0;
                    }
                    else
                    {
                        std::cerr << "undefined CellPDDefault" << std::endl;
                        exit(-1);
                    }
                }

                //std::cout << energy.CellPD << std::endl;
            }

            if (energy.computeUseProbability)
            {
                if (config["energy_cal"]["inVPD"])
                {
                    energy.inVPD = config["energy_cal"]["inVPD"].as<std::vector<double>>();
                    double sum = 0;
                    for (auto x:energy.inVPD)
                    {
                        sum += x;
                    }

                    double epsilon = 1e-5;
                    if (energy.inVPD.size() != inVLevels)
                    {
                        std::cerr << "inVPD illegal: the number of probabilities is not inVLevels" << std::endl;
                        exit(-1);
                    }
                    else if (fabs(sum - 1.0) > epsilon)
                    {
                        std::cerr << "inVPD illegal: the sum of probabilities is not 1" << std::endl;
                        exit(-1);
                    }
                }
                else
                {
                    energy.inVPDDefault = config["energy_cal"]["inVPDDefault"].as<int>();
                    if (energy.inVPDDefault == 0)
                    {
                        energy.inVPD = std::vector<double>(inVLevels, 1.0/inVLevels);
                    }
                    else if (energy.inVPDDefault == 1)
                    {
                        energy.inVPD = std::vector<double>(inVLevels, 0);
                        energy.inVPD[inVLevels - 1] = 1.0;
                    }
                    else
                    {
                        std::cerr << "undefined inVPDDefault" << std::endl;
                        exit(-1);
                    }
                }

                //std::cout << energy.inVPD << std::endl;
            }

            //calculate averageEnergyPerWrite
            {
                energy.averageEnergyPerWrite = std::vector<double>(energy.writeParallelism + 1);
                double VoltageSquareMulTime = writeV/2 * writeV/2 * lat_area.phyWrLatency;
                double conductanceSum = 0;
                double average_conductance = 0;

                for (int i = 0; i < energy.CellPD.size(); ++i)
                {
                    average_conductance += i * energy.CellPD[i];
                }
                double deltaConduct = (maxConduct - minConduct) / (cellLevels - 1);
                average_conductance *= deltaConduct;
                average_conductance += minConduct;

                //i cells in every writeParallelism cells need to write
                for (int i = 0; i <= energy.writeParallelism; ++i)
                {
                    conductanceSum = 0;
                    if (i)
                    {
                        //half selected row
                        conductanceSum += (phyArrColSize - i) * average_conductance;
                        //half selected col
                        conductanceSum += i * (phyArrRowSize - 1) * average_conductance;
                        //full selected cells
                        conductanceSum += i * average_conductance * 4;

                        //ref column need extra two cells
                        if (mode == 1)
                        {
                            conductanceSum += 2 * average_conductance;
                        }
                    }

                    energy.averageEnergyPerWrite[i] = VoltageSquareMulTime * conductanceSum;
                }
                std::cout << "energy.averageEnergyPerWrite: " << energy.averageEnergyPerWrite << std::endl;

            }
        }
      
        //ir_drop
        ir_drop.enable = config["ir_drop"]["enable"].as<bool>();
        if (ir_drop.enable)
        {
            ir_drop.g_wire = config["ir_drop"]["g_load"].as<double>();
            ir_drop.g_load = config["ir_drop"]["g_wire"].as<double>();
            ir_drop.fast_mode = config["ir_drop"]["fast_mode"].as<bool>();
            ir_drop.times = config["ir_drop"]["times"].as<int>();
            if (ir_drop.fast_mode && inVBits>1 && has_negative_input)
            {
                TORCH_CHECK(false, "ir drop fast mode don't support negative voltage inputs\r\n");
            }
        }
        //stuck at fault
        sa.enable = config["SAF"]["enable"].as<bool>();
        if (sa.enable)
        {
            sa.init_pSA0 = config["SAF"]["init_pSA0"].as<double>();
            sa.init_pSA1 = config["SAF"]["init_pSA1"].as<double>();
            sa.runtime_en = config["SAF"]["runtime_en"].as<bool>();
            sa.pSA0 = config["SAF"]["pSA0"].as<double>();
            sa.pSA1 = config["SAF"]["pSA1"].as<double>();
        }

    }
};

/**
 * Generate the only default pim array pro config.
 * */
const pim_array_pro_config& pro_decf()
{
    static const pim_array_pro_config decf(PIM_ARRAY_CONFIG_PATH);
    return decf;
}

#endif //PIMTORCH_PIM_ARRAY_CONFIG_H
