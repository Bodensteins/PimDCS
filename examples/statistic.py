import torch.nn as nn
from src.pimtorch.nn.modules.conv_ou import Conv2d_OU
from src.pimtorch.nn.modules.linear_ou import Linear_OU
from src.pimtorch.nn.fixedPointDataAnalyze import FpDataAnalyzer_PN

def print_nn_Sequential_Modules_statistic(modules:nn.Sequential):
    input_counts_dict = {}
    # weight_count_dict_pos = {}
    # weight_count_dict_neg = {}
    ou_out_count_dict = {}
    for module in modules():
        if isinstance(module, Conv2d_OU) or isinstance(module, Linear_OU):
            if module.mm_manager_type == 0:
                continue

            print("---------------------")
            print(module)
            module.print_statistic()
            print("---------------------\n")

            # 将所有层的bit vec ones num数据加在一起
            input_ones_num_dict = module.data_analyzer.input_bit_vec_ones_num_dict
            for ones in input_ones_num_dict:
                if input_counts_dict.__contains__(ones):
                    input_counts_dict[ones] += input_ones_num_dict[ones]
                else:
                    input_counts_dict[ones] = input_ones_num_dict[ones]
            
            # weight_ones_num_dict_pos = None
            # weight_ones_num_dict_neg = None
            ou_out_dict = None
            if isinstance(module.data_analyzer, FpDataAnalyzer_PN):
                # weight_ones_num_dict_pos = module.data_analyzer.weight_bit_vec_ones_num_dict_pos
                # for ones in weight_ones_num_dict_pos:
                #     if weight_count_dict_pos.__contains__(ones):
                #         weight_count_dict_pos[ones] += weight_ones_num_dict_pos[ones]
                #     else:
                #         weight_count_dict_pos[ones] = weight_ones_num_dict_pos[ones]
                # weight_ones_num_dict_neg = module.data_analyzer.weight_bit_vec_ones_num_dict_neg
                # for ones in weight_ones_num_dict_neg:
                #     if  weight_count_dict_neg.__contains__(ones):
                #         weight_count_dict_neg[ones] += weight_ones_num_dict_neg[ones]
                #     else:
                #         weight_count_dict_neg[ones] = weight_ones_num_dict_neg[ones]
                ou_out_dict = module.data_analyzer.ou_column_output_num_dict
                for value in ou_out_dict:
                    if ou_out_count_dict.__contains__(value):
                        ou_out_count_dict[value] += ou_out_dict[value]
                    else:
                        ou_out_count_dict[value] = ou_out_dict[value]

    # 排序
    ou_out_count_dict = dict(sorted(ou_out_count_dict.items()))

    print("\n----------total-----------")
    print("\n不同1数量的输入向量的出现次数和频率:")
    total_counts = sum(input_counts_dict.values())
    # print(input_counts_dict)
    for ones_num in input_counts_dict:
        counts = input_counts_dict[ones_num]
        propotion = counts / total_counts
        if ones_num != 0:
            non_zero_propotion = counts / (total_counts - input_counts_dict[0])
        else:
            non_zero_propotion = 0
        print(f"ones num:{ones_num}, counts:{counts}, propotion:{propotion:.2%}, non zero propotion:{non_zero_propotion:.2%}")

    # print("\n不同1数量的OU列的出现次数和频率:")
    # total_counts_pos = sum(weight_count_dict_pos.values())
    # # print(input_counts_dict)
    # for ones_num in weight_count_dict_pos:
    #     counts = weight_count_dict_pos[ones_num]
    #     propotion = counts / total_counts_pos
    #     print(f"positive -- ones num:{ones_num}, counts:{counts}, propotion:{propotion:.2%}")
    # total_counts_neg = sum(weight_count_dict_neg.values())
    # for ones_num in weight_count_dict_neg:
    #     counts = weight_count_dict_neg[ones_num]
    #     propotion = counts / total_counts_neg
    #     print(f"negative -- ones num:{ones_num}, counts:{counts}, propotion:{propotion:.2%}")
        
    print("\n不同OU输出结果的出现次数和频率(正负OU结果相减):")
    total_counts = sum(ou_out_count_dict.values())
    # print(input_counts_dict)
    for value in ou_out_count_dict:
        counts = ou_out_count_dict[value]
        propotion = counts / total_counts
        if value != 0:
            non_zero_propotion = counts / (total_counts - ou_out_count_dict[0])
        else:
            non_zero_propotion = 0
        print(f"ou out value:{value}, counts:{counts}, propotion:{propotion:.2%}, non zero propotion:{non_zero_propotion:.2%}")