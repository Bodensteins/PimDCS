import math

def add_to_dict(key, value, data_dict):
    if data_dict.__contains__(key):
        data_dict[key] += value
    else:
        data_dict[key] = value

def merge_dict(dict0, dict1):
    new_dict = {}
    for key in dict0:
        add_to_dict(key, dict0[key], new_dict)
    for key in dict1:
        add_to_dict(key, dict1[key], new_dict)
    return new_dict

def twin_range_quantize_sar(data_dict, nr1, nr2, bias):
    interval_a = 1
    num_AD_read = sum(data_dict.values())
    R1_max = 2 ** nr1
    threshold = interval_a * (R1_max - 1)
    sample_in_R1 = 0
    for val in data_dict:
        if val < threshold + bias:
            sample_in_R1 += data_dict[val]
    num_AD_ops = num_AD_read + sample_in_R1 * nr1 + (num_AD_read - sample_in_R1) * nr2
    return num_AD_ops

def merge_input_data_dict_list(data_dict_list, input_slice_num, weight_slice_num):
    new_data_dict_list = []
    ou_row_num = len(data_dict_list[0][0][0])
    ou_col_num = len(data_dict_list[0][0][0][0])

    for s in range(2):
        new_data_dict_list.append([])
        for w in range(weight_slice_num):
            new_data_dict_list[s].append([])

    for s in range(2):
        for i in range(input_slice_num):
            for w in range(weight_slice_num):
                for r in range(ou_row_num):
                    for c in range(ou_col_num):
                        data_dict = data_dict_list[s][i][w][r][c]
                        if len(data_dict) == 0:
                            continue
                        new_data_dict_list[s][w] = merge_dict(new_data_dict_list[s][w], data_dict)

    return new_data_dict_list

def merge_all_data_dict_list_sample(data_dict_list, weight_slice_num):
    new_data_dict_list = []

    for s in range(2):
        new_data_dict_list.append([])

    for s in range(2):
            for w in range(weight_slice_num):
                data_dict = data_dict_list[s][w]
                if len(data_dict) == 0:
                    continue
                new_data_dict_list[s] = merge_dict(new_data_dict_list[s], data_dict)

    return new_data_dict_list

def merge_all_data_dict_list_test(data_dict_list, input_slice_num, weight_slice_num):
    new_data_dict_list = []
    ou_row_num = len(data_dict_list[0][0][0])
    ou_col_num = len(data_dict_list[0][0][0][0])

    for s in range(2):
        new_data_dict_list.append([])

    for s in range(2):
        for i in range(input_slice_num):
            for w in range(weight_slice_num):
                for r in range(ou_row_num):
                    for c in range(ou_col_num):
                        data_dict = data_dict_list[s][i][w][r][c]
                        if len(data_dict) == 0:
                            continue
                        new_data_dict_list[s] = merge_dict(new_data_dict_list[s], data_dict)

    return new_data_dict_list

def trq_caibration(data_dict, resolution):
    minimum_v = 1e-12
    nr1 = 0
    nr2 = 0
    bias = 0

    x_range = max(data_dict) + minimum_v

    max_similarity=-1e9

    alpha = 0.1
    beta = 1.2
    num_AD_read = sum(data_dict.values())
    lossless_max_res = math.ceil(math.log2(x_range + 1))
    lossless_max_res = min(lossless_max_res, resolution)

    if lossless_max_res == 0:
        nr1 = 0
        nr2 = 0
        bias = 0
    # Only 1 candidate
    elif lossless_max_res == 1:
        nr1 = 1
        nr2 = 1
        bias = 0
    # Only 2 candidate
    else:
        prev_lossless_R2_res = -1
        for Range_2 in range(math.ceil(x_range * alpha), math.ceil(x_range * beta)):
            lossless_R2_res = math.ceil(math.log2(Range_2 + 1))
            if prev_lossless_R2_res == lossless_R2_res:
                continue
            prev_lossless_R2_res = lossless_R2_res
            # re-constrain the bit requirement for R2, beacuse Range_2 > x_range
            N_R2 = min(lossless_R2_res, resolution)
            # Quantization levels for R2
            R2_max = 2 ** N_R2
            interval_b = Range_2 / (R2_max - 1)

            min_num_AD_ops = 1e15
            best_bias = None
            for m in range(1, lossless_max_res):                         
                # calculate AD ops
                N_R1 = lossless_R2_res - m 
                R1_max = 2 ** N_R1
                interval_a = 1
                threshold = interval_a * (R1_max-1)
                bias = 0
                sample_in_R1 = 0
                for val in data_dict:
                    if val < threshold + bias:
                        sample_in_R1 += data_dict[val]
                num_AD_ops = num_AD_read + sample_in_R1 * N_R1 + (num_AD_read - sample_in_R1) * N_R2

                if num_AD_ops < min_num_AD_ops: 
                    min_num_AD_ops = num_AD_ops
                    best_m = m
                    best_bias = bias
                    
            m = best_m
            lossless_R1_res = lossless_R2_res - m
            N_R1 = min(lossless_R1_res,resolution)
            R1_max = 2**N_R1
            interval_a = 1
            threshold = interval_a * (R1_max-1)

            similarity = 0
            for val in data_dict:
                if val < threshold + bias:
                    val_q = val
                else:
                    val_q = round(val / interval_b)
                    if val > R2_max - 1:
                        val_q = R2_max - 1
                    val_q *= interval_b
                similarity -= (val - val_q) * (val - val_q) * data_dict[val]
            # print(N_R1, N_R2, m, threshold, bias, similarity)

            if similarity > max_similarity:
                max_similarity = similarity
                nr1 = N_R1
                nr2 = N_R2
                bias = best_bias
    return nr1, nr2, bias

def cal_layer_cmp_times(dict_list_sample, dict_list_test, resolution):
    input_slice_num = len(dict_list_test[0])
    weight_slice_num = len(dict_list_sample[0])
    trq_para_list_dict = {}
    total_trq_cmp_times = 0

    trq_dict_list_test = merge_input_data_dict_list(dict_list_test, input_slice_num, weight_slice_num)

    for sign in range(2):
        for w in range(weight_slice_num):
            data_dict_sample = dict_list_sample[sign][w]
            data_dict_test = trq_dict_list_test[sign][w]

            trq_nr1, trq_nr2, trq_bias = trq_caibration(data_dict_sample, resolution)

            trq_str = str(trq_nr1) + '_' + str(trq_nr2)
            add_to_dict(trq_str, 1, trq_para_list_dict)
            
            trq_cmp_times = twin_range_quantize_sar(data_dict_test, trq_nr1, trq_nr2, trq_bias)
            total_trq_cmp_times += trq_cmp_times

    return total_trq_cmp_times, trq_para_list_dict

