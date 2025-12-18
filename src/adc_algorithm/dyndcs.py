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


def data_list_to_dict(data_list, in_slice_num=8, w_slice_num=4, adc_bl_share=128):
    r_len = len(data_list[0][0])
    c_len = len(data_list[0][0][0])

    data_dict_list = []

    for s in range(2):
        data_dict_list.append([])
        for i in range(in_slice_num):
            data_dict_list[s].append([])
            for w in range(w_slice_num):
                data_dict_list[s][i].append([])
                for r in range(r_len):
                    data_dict_list[s][i][w].append([])
                    for c in range(c_len):
                        data_dict_list[s][i][w][r].append({})

    for s in range(2):
        for w in range(w_slice_num):
            for r in range(r_len):
                for c in range(c_len):
                    d_list = data_list[s][w][r][c]
                    for j, d in enumerate(d_list):
                        # print(d)
                        i = (j // adc_bl_share) % in_slice_num
                        data_dict = data_dict_list[s][i][w][r][c]
                        add_to_dict(d, 1, data_dict)

    return data_dict_list


def convert(data, resolution, ns_start, noff_start):
    assert ns_start > noff_start

    ns = ns_start
    noff = noff_start
    conv_step_num = 0
    cur_state = 0
    cur_bit = noff
    err_state = 0
    sref = 2 ** ns - 2 ** noff
    print(f"\ndata={data}({format(data, '0'+str(resolution)+'b')}), reolution={resolution}")
    while True:
        print(f"conv_step={conv_step_num}, ns={ns}, noff={noff}, sref={sref}({format(sref, '0'+str(resolution)+'b')}), cur_state={cur_state}, cur_bit={cur_bit}, err_state={err_state}")
        conv_step_num += 1
        if cur_state == 0:
            if data >= sref:
                cur_state = 1
                cur_bit = ns
                sref = 2 ** ns
            else:
                if ns > noff + 1:
                    cur_state = 2
                    cur_bit = noff
                    noff += 1
                    sref = 2 ** ns - 2 ** noff
                else:
                    cur_state = 3
                    noff -= 1
                    cur_bit = noff
                    sref = 2 ** noff
        elif cur_state == 1:
            if data >= sref:
                if ns == resolution - 1: # worst
                    cur_state = -1
                    cur_bit = resolution - 2
                else:    # x1
                    cur_state = 4
                    err_state = 1
                    ns += 1
                    cur_bit = ns
                    sref = 2 ** ns
            else:   # o2
                cur_state = -1  # 5
                noff -= 1
                cur_bit = noff
                if noff_start > 0:
                    err_state = 4
                sref = 2 ** ns - 2 ** noff
        elif cur_state == 2:
            if data >= sref:
                cur_state = -1
                cur_bit -= 1
                sref = 2 ** ns - 2 ** noff + 2 ** (noff - 2)
            else:   # x2
                err_state = 3
                if ns > noff + 1:
                    cur_bit += 1
                    noff += 1
                    sref = 2 ** ns - 2 ** noff
                else:
                    cur_state = 3   # else state
                    noff -= 1
                    cur_bit = noff
                    sref = 2 ** noff
        elif cur_state == 3:   # o1
            cur_state = -1
            cur_bit -= 1
            if err_state == 0:
                err_state = 2
            if data >= sref:
                sref = 2 ** noff + 2 ** (noff - 1)
            else:
                sref = 2 ** (noff - 1)
        elif cur_state == 4:
            if data >= sref:
                if ns == resolution - 1: # worst
                    cur_state = -1
                    cur_bit = resolution - 2
                    sref = 2 ** (resolution - 1) + 2 ** (resolution - 2)
                else:
                    ns += 1
                    cur_bit = ns
                    sref = 2 ** ns
            else:
                ns -= 2
                cur_state = -1
                cur_bit = ns
                sref = 2 ** ns + 2 ** (ns + 1)
        else: # BS
            if cur_bit == -1:
                print(f"cur_state:{cur_state}")
            conv_step_num += cur_bit
            cur_bit = -1
            # print(conv_step_num)

        if cur_bit == -1:
            break    
    
    return conv_step_num, err_state


def dynamic_dcs(data_list, resolution, ns_start, noff_start, max_err_cnt=3):
    assert ns_start > noff_start and resolution >= ns_start and noff_start >= 0
    
    err_state = 0
    err_cnt = 0
    total_conv_step_num = 0
    ns = ns_start
    noff = noff_start

    conv_step_dict = {}

    for d in data_list:
        conv_step_num, new_err_state = convert(d, resolution, ns, noff)
        total_conv_step_num += conv_step_num
        add_to_dict(conv_step_num, 1, conv_step_dict)
        if new_err_state != err_state:
            err_state = new_err_state
            err_cnt = 0
        else:
            err_cnt += 1
            if err_cnt == max_err_cnt:
                if err_state == 1:
                    ns += 1
                    noff = ns - 1
                elif err_state == 2:
                    noff -= 1
                    ns = noff + 1
                elif err_state == 3:
                    noff += 1
                elif err_state == 4:
                    noff -= 1
                err_cnt = 0
    
    return total_conv_step_num, conv_step_dict


def cal_layer_conv_step_nums(test_cycle_list, resolution, max_err_cnt=2):
    total_conv_step_num = 0
    total_conv_step_dict = {}
    
    ns_start = 4
    noff_start = 3

    for s_list in test_cycle_list:
        for w_list in s_list:
            for r_list in w_list:
                for c_list in r_list:
                    conv_step_num, conv_step_dict = dynamic_dcs(c_list, resolution, ns_start, noff_start, max_err_cnt)
                    total_conv_step_num += conv_step_num
                    total_conv_step_dict = merge_dict(total_conv_step_dict, conv_step_dict)

    return total_conv_step_num, total_conv_step_dict

