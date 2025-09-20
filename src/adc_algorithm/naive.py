def naive_sar(data_dict, resolution):
    return sum(data_dict.values()) * resolution

def cal_layer_cmp_times(data_dict_list, resolution):
    input_slice_num = len(data_dict_list[0])
    weight_slice_num = len(data_dict_list[0][0])
    xbar_row_num = len(data_dict_list[0][0][0])
    xbar_col_num = len(data_dict_list[0][0][0][0])
    
    total_cmp_times = 0

    for s in range(2):
        for i in range(input_slice_num):
            for w in range(weight_slice_num):
                for r in range(xbar_row_num):
                    for c in range(xbar_col_num):
                        cmp_times = naive_sar(data_dict_list[s][i][w][r][c], resolution)
                        total_cmp_times += cmp_times

    return total_cmp_times
