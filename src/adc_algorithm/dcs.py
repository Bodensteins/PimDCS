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

def find_distribution_type(data_dict:dict):
    if len(data_dict) <= 1:
        return 'monitonic'
    elif 0 not in data_dict or 1 not in data_dict:
        return 'gaussian'
    max_val = max(data_dict)
    for i in range(1, max_val):
        if data_dict.__contains__(i):
            if data_dict[i] > data_dict[0]:
                return 'gaussian'
    return 'monotonic'

def gaus_convert(d, resolution, n_start, n_step, n_off):
    if n_start == resolution - 1:
        return resolution
    vref_start = (1 << n_start) - (1 << n_off)
    cmp_times = 1
    if d < vref_start:
        # n_off不受n_step控制，始终只回退1步
        for i, n in enumerate(range(n_off + 1, n_start)):
            vref_pred = (1 << n_start) - (1 << n)
            if d >= vref_pred:
                cmp_times += i + n + 1
                break
        if cmp_times == 1:
            cmp_times += 2 * n_start - n_off - 2
    else:
        vref_pred = 1 << n_start
        cmp_times += 1
        if d < vref_pred:
            cmp_times += n_off
        else:
            for i, n in enumerate(range(n_start + n_step, resolution, n_step)):
                vref_pred = 1 << n
                if d < vref_pred:
                    cmp_times += i + n + 1
                    if n_step == 1:
                        cmp_times -= 1
                    break
            if cmp_times == 2:
                additional_cmp_times = math.ceil((resolution - n_start - n_step) / n_step)
                cmp_times = resolution + additional_cmp_times
                if n_step == 1 and d < 1 << (resolution - 1):
                    cmp_times -= 1
    return cmp_times

def mono_convert(d, resolution, n_start, n_step):
    if n_start == resolution - 1:
        return resolution
    cmp_times = 0
    for i, n_vref in enumerate(range(n_start, resolution, n_step)):
        vref_pred = 1 << n_vref
        if d < vref_pred:
            cmp_times = i + n_vref + 1
            if n_step == 1 and i > 0:
                cmp_times -= 1
            break
    if cmp_times == 0:
        additional_cmp_times = math.ceil((resolution - n_start) / n_step)
        cmp_times = resolution + additional_cmp_times
        if n_step == 1 and d < 1 << (resolution - 1):
            cmp_times -= 1
    return cmp_times

def gaus_para_search(data_dict, resolution, threshold):
    best_n_start = resolution - 1
    best_n_step = 1
    best_n_off = 0
    min_cmp_times = 1e14
    for n_start in range(resolution):
        for n_step in range(1, resolution - n_start):
            for n_off in range(n_start):
                cmp_times = 0
                threshold_flag = False
                for d in data_dict:
                    temp_times = gaus_convert(d, resolution, n_start, n_step, n_off)
                    if temp_times > threshold:
                        threshold_flag = True
                        break
                    cmp_times += temp_times * data_dict[d]
                if threshold_flag:
                    continue
                if cmp_times < min_cmp_times:
                    best_n_start = n_start
                    best_n_step = n_step
                    best_n_off = n_off
                    min_cmp_times = cmp_times
                # print(n_start, n_step, n_off, cmp_times)
    return best_n_start, best_n_step, best_n_off, min_cmp_times

def mono_para_search(data_dict, resolution, threshold):
    best_n_start = resolution - 1
    best_n_step = 1
    min_cmp_times = 1e14
    for n_start in range(resolution):
        for n_step in range(1, resolution - n_start):
            cmp_times = 0
            threshold_flag = False
            for d in data_dict:
                temp_times = mono_convert(d, resolution, n_start, n_step)
                if temp_times > threshold:
                    threshold_flag = True
                    break
                cmp_times += temp_times * data_dict[d]
            if threshold_flag:
                continue
            if cmp_times < min_cmp_times:
                min_cmp_times = cmp_times
                best_n_start = n_start
                best_n_step = n_step
    return best_n_start, best_n_step, min_cmp_times

def para_search_diff(data_dict, resolution, threshold=16):
    distribution_type = find_distribution_type(data_dict)
    n_start = 0
    n_step = 1
    n_off = 0
    flag = 0
    min_cmp_times = 0
    if distribution_type == 'gaussian':
        flag = 1
        n_start, n_step, n_off, min_cmp_times = gaus_para_search(data_dict, resolution, threshold)
    else:
        flag = 0
        n_start, n_step, min_cmp_times = mono_para_search(data_dict, resolution, threshold)
    return n_start, n_step, n_off, flag, min_cmp_times

def distribution_aware_para_search(xbar_data_dict_list, resolution, threshold):
    mono_dict = {}
    gaus_dict = {}
    
    c0 = 0
    c1 = 0
    prev_type = 'gaussian'
    for data_dict in xbar_data_dict_list:
        if prev_type == 'monotonic':
            c1 += 1
            mono_dict = merge_dict(data_dict, mono_dict)
        elif find_distribution_type(data_dict) == 'gaussian':
            c0 += 1
            prev_type = 'gaussian'
            gaus_dict = merge_dict(data_dict, gaus_dict)
        else:
            c1 += 1
            prev_type = 'monotonic'
            mono_dict = merge_dict(data_dict, mono_dict)
    n_start_0 = 0
    n_step_0 = 1
    n_off_0 = 0
    if c0 != 0:
        n_start_0, n_step_0, n_off_0, _ = gaus_para_search(gaus_dict, resolution, threshold)

    n_start_1 = 0
    n_step_1 = 1
    if c1 != 0:
        n_start_1, n_step_1, _ = mono_para_search(mono_dict, resolution, threshold)

    return [c0, c1, n_start_0, n_step_0, n_off_0, n_start_1, n_step_1]

def distribution_aware_convert(xbar_data_dict_list, resolution, para_list):
    total_cmp_times = 0
    cmp_times_dict = {}

    c0 = para_list[0]
    c1 = para_list[1]
    assert len(xbar_data_dict_list) == c0 + c1

    n_start_0 = para_list[2]
    n_step_0 = para_list[3]
    n_off_0 = para_list[4]
    if c0 != 0:
        for i in range(c0):
            data_dict = xbar_data_dict_list[i]
            for d in data_dict:
                cmp_times = gaus_convert(d, resolution, n_start_0, n_step_0, n_off_0)
                add_to_dict(cmp_times, data_dict[d], cmp_times_dict)
                total_cmp_times += cmp_times * data_dict[d]
    
    n_start_1 = para_list[5]
    n_step_1 = para_list[6]
    if c1 != 0:
        for i in range(c0, c0 + c1):
            data_dict = xbar_data_dict_list[i]
            for d in data_dict:
                cmp_times = mono_convert(d, resolution, n_start_1, n_step_1)
                add_to_dict(cmp_times, data_dict[d], cmp_times_dict)
                total_cmp_times += cmp_times * data_dict[d]
    
    return total_cmp_times, cmp_times_dict

def cal_layer_cmp_times(dict_list_sample, dict_list_test, resolution, threshold):
    assert threshold >= resolution

    input_slice_num = len(dict_list_sample[0])
    weight_slice_num = len(dict_list_sample[0][0])
    xbar_row_num = len(dict_list_sample[0][0][0])
    xbar_col_num = len(dict_list_sample[0][0][0][0])
    
    para_list_dict = {}
    total_cmp_times_dict = {}
    total_cmp_times = 0

    for s in range(2):
        for w in range(weight_slice_num):
            for r in range(xbar_row_num):
                for c in range(xbar_col_num):
                    xbar_data_dict_list_sample = []
                    xbar_data_dict_list_test = []
                    for i in range(input_slice_num):
                        xbar_data_dict_list_sample.append(dict_list_sample[s][i][w][r][c])
                        xbar_data_dict_list_test.append(dict_list_test[s][i][w][r][c])

                    para_list = distribution_aware_para_search(xbar_data_dict_list_sample, resolution, threshold)
                    cmp_times, cmp_times_dict= distribution_aware_convert(xbar_data_dict_list_test, resolution, para_list)

                    para_list_str = str(para_list)
                    add_to_dict(para_list_str, 1, para_list_dict)

                    total_cmp_times_dict = merge_dict(total_cmp_times_dict, cmp_times_dict)
                    total_cmp_times += cmp_times

    return total_cmp_times, total_cmp_times_dict, para_list_dict
    