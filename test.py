from src.adc_algorithm.adc_algorithm_test import nn_test_final
from src.adc_algorithm import dyndcs, dcs, trq, naive
from temp.test2 import read_input_dict_list, read_cycle_list, read_train_dict_list, read_dict_list
import pickle

def test_convert():
    resolution = 8
    data = [0, 1, 2, 26, 29, 28, 17, 4, 22, 36, 111, 245, 17, 14, 11, 7, 1, 12, 63, 221, 4]
    ns = [1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 7, 7, 7]
    noff = [0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 6, 6, 6]
    for d, s, o in zip(data, ns, noff):
        conv_step_num, err_state = dyndcs.convert(d, resolution, s, o)
        print(f"\nconv_step_num={conv_step_num}, err_state={err_state}")
        print("------------------------------------")


def test_dyndcs():
    data_list = [29, 25, 12, 13, 21, 21, 24, 15, 20, 16, 18]
    # data_list = [0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 1]
    resolution = 8
    ns_start = 5
    noff_start = 4
    max_err_cnt = 2
    total_conv_step_num_dcs = dyndcs.dynamic_dcs(data_list, resolution, ns_start, noff_start, max_err_cnt)
    total_conv_step_num_naive = len(data_list) * resolution
    print(f"\n{total_conv_step_num_dcs / total_conv_step_num_naive : .2%}")


def dynamic_dcs_debug(data_list, resolution, ns_start, noff_start, max_err_cnt=3):
    assert ns_start > noff_start and resolution >= ns_start and noff_start >= 0
    
    err_state = 0
    err_cnt = 0
    total_conv_step_num = 0
    ns = ns_start
    noff = noff_start
    slice_conv_step_num = 0

    # data_dict = {}

    for i, d in enumerate(data_list):
        conv_step_num, new_err_state = dyndcs.convert(d, resolution, ns, noff)
        slice_conv_step_num += conv_step_num
        total_conv_step_num += conv_step_num
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
        
        # if data_dict.__contains__(d):
        #     data_dict[d] += 1
        # else:
        #     data_dict[d] = 1

        if (i + 1) % 128 == 0:
            print(f"in_bit={i // 128}")
            print(f"dyndcs to naive: {slice_conv_step_num / (128 * resolution): .4%}")
            slice_conv_step_num = 0

        #     data_dict = dict(sorted(data_dict.items()))
        #     total_counts = sum(data_dict.values())
        #     for d in data_dict:
        #         counts = data_dict[d]
        #         propotion = counts / total_counts
        #         print(f"data:{d}, counts:{counts}, propotion:{propotion:.2%}")

        #     data_dict = {}

        if i == 1023:
            break
        # print(f"\nerr_stat={err_state}, err_cnt={err_cnt}, ns={ns}, noff={noff}")
        # print("------------------------------------")
    
    return total_conv_step_num


def test_dyndcs_xbr():
    with open('/home/leitaoming/Data/FADESim_data/exp_data/resnet18_imagenet/layer7_spec_xbr128_cycle.pkl', 'rb') as f:
        output_cycle_list = pickle.load(f)

    with open('/home/leitaoming/Data/FADESim_data/exp_data/resnet18_imagenet/layer7_spec_xbr128_cellbit2_sample_xbr_output.pkl', 'rb') as f:
        spec_sample_list = pickle.load(f)

    resolution = 8
    ns_start = [1, 1, 7, 7]
    noff_start = [0, 0, 6, 6]
    max_err_cnt = 2
    total_conv_step_num_dcs = 0
    total_conv_step_num_naive = 0

    for i, data_list in enumerate(output_cycle_list):
        print(f"\nw_bit={i}")
        conv_step_num_dcs = dynamic_dcs_debug(data_list, resolution, ns_start[i], noff_start[i], max_err_cnt)
        conv_step_num_naive = 1024 * resolution
        total_conv_step_num_dcs += conv_step_num_dcs
        total_conv_step_num_naive += conv_step_num_naive
        print(f"\nxbr{i}: {conv_step_num_dcs / conv_step_num_naive : .4%}")
    
    print(f"\ntotal: {total_conv_step_num_dcs / total_conv_step_num_naive : .4%}")

    para_list = []
    for w in range(4):
        xbar_data_dict_list_sample = []
        for i in range(8):
            xbar_data_dict_list_sample.append(spec_sample_list[0][i][w][0][0])
        para = dcs.distribution_aware_para_search(xbar_data_dict_list_sample, resolution, 14)
        para_list.append(para)
    print(para_list)
        
    total_conv_step_num = 0
    for w in range(4):
        print(f"\nw_bit={w}")
        c0, c1, n_start_0, n_step_0, n_off_0, n_start_1, n_step_1 = para_list[w]
        slice_conv_step_num = 0
        xbr_conv_step_num = 0
        
        for i, d in enumerate(output_cycle_list[w]):
            c = i // 128
            if c < c0:
                step_num = dcs.gaus_convert(d, resolution, n_start_0, n_step_0, n_off_0)
            else:
                step_num = dcs.mono_convert(d, resolution, n_start_1, n_step_1)
            slice_conv_step_num += step_num
            xbr_conv_step_num += step_num
            total_conv_step_num += step_num

            if (i + 1) % 128 == 0:
                print(f"in_bit={c}")
                print(f"stdcs to naive: {slice_conv_step_num / (128 * resolution): .4%}")
                slice_conv_step_num = 0

            if i == 1023:
                break

        print(f"\nxbr{w}: {xbr_conv_step_num / conv_step_num_naive : .4%}")

    print(f"\ntotal: {total_conv_step_num / total_conv_step_num_naive : .4%}")


# alexnet
# vgg11
# vgg16
# resnet18
def test_dyndcs_full():
    data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/alexnet_cifar10/', 7
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/vgg11_cifar10/', 11
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/vgg16_imagenet/', 16
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/resnet18_imagenet/', 21

    resolution = 8
    dyndcs_max_err_cnt = 2
    dcs_conv_step_num_threshold = 16

    in_slice_num = 8
    w_slice_num = 4
    adc_bl_share = 128

    total_dyndcs_conv_step_num = 0
    total_dcs_conv_step_num = 0
    total_naive_conv_step_num = 0
    total_trq_conv_step_num = 0

    dyndcs_conv_step_dict = {}
    # spec_para_list_dict = {}
    # trq_para_list_dict = {}

    for layer_no in range(layer_num):
        sample_file = '/layer' + str(layer_no) + '_spec_xbr128_cellbit2_sample_xbr_output.pkl' 
        trq_sample_file = '/layer' + str(layer_no) + '_trq_xbr128_cellbit2_test_xbr_output.pkl'
        test_file = '/layer' + str(layer_no) + '_spec_xbr128_cycle.pkl'

        try:
            # with open(data_path+sample_file, 'rb') as f:
            #     sample_data_dict_list = pickle.load(f)
            with open(data_path+trq_sample_file, 'rb') as f:
                trq_data_dict_list = pickle.load(f)
            with open(data_path+test_file, 'rb') as f:
                test_data_cycle_list = pickle.load(f)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Unexcept error:{e}")

        test_data_dict_list = dyndcs.data_list_to_dict(test_data_cycle_list, in_slice_num, w_slice_num, adc_bl_share)

        layer_dyndcs_conv_step_num, layer_dyndcs_conv_step_dict = dyndcs.cal_layer_conv_step_nums(test_data_cycle_list, resolution, dyndcs_max_err_cnt)
        layer_dcs_conv_step_num, _, _ = dcs.cal_layer_cmp_times(test_data_dict_list, test_data_dict_list, resolution, dcs_conv_step_num_threshold)
        layer_trq_conv_step_num, _ = trq.cal_layer_cmp_times(trq_data_dict_list, test_data_dict_list, resolution)
        layer_naive_conv_step_num = naive.cal_layer_cmp_times(test_data_dict_list, resolution)

        total_dyndcs_conv_step_num += layer_dyndcs_conv_step_num
        total_dcs_conv_step_num += layer_dcs_conv_step_num
        total_trq_conv_step_num += layer_trq_conv_step_num
        total_naive_conv_step_num += layer_naive_conv_step_num

        dyndcs_conv_step_dict = dyndcs.merge_dict(dyndcs_conv_step_dict, layer_dyndcs_conv_step_dict)

        print(f"\nlayer_no={layer_no}")
        print(f"dyndcs_conv_step_num:{layer_dyndcs_conv_step_num}")
        print(f"dcs_conv_step_num:{layer_dcs_conv_step_num}")
        print(f"naive_conv_step_num:{layer_naive_conv_step_num}")
        print(f"trq_conv_step_num:{layer_trq_conv_step_num}")
        print(f"dyndcs to naive:{layer_dyndcs_conv_step_num / layer_naive_conv_step_num : .2%}")
        print(f"dcs to naive:{layer_dcs_conv_step_num / layer_naive_conv_step_num : .2%}")
        print(f"trq to naive:{layer_trq_conv_step_num / layer_naive_conv_step_num : .2%}")
        print(f"dyndcs to trq:{layer_dyndcs_conv_step_num / layer_trq_conv_step_num : .2%}")
        print(f"dcs to trq:{layer_dcs_conv_step_num / layer_trq_conv_step_num : .2%}")

    print("\n------------------------------------------------\n")
    print(f"total_dyndcs_conv_step_num:{total_dyndcs_conv_step_num}")
    print(f"total_dcs_conv_step_num:{total_dcs_conv_step_num}")
    print(f"total_naive_conv_step_num:{total_naive_conv_step_num}")
    print(f"total_trq_conv_step_num:{total_trq_conv_step_num}")
    print(f"total dyndcs to naive:{total_dyndcs_conv_step_num / total_naive_conv_step_num : .2%}")
    print(f"total dcs to naive:{total_dcs_conv_step_num / total_naive_conv_step_num : .2%}")
    print(f"total trq to naive:{total_trq_conv_step_num / total_naive_conv_step_num : .2%}")
    print(f"total dyndcs to trq:{total_dyndcs_conv_step_num / total_trq_conv_step_num : .2%}")
    print(f"total dcs to trq:{total_dcs_conv_step_num / total_trq_conv_step_num : .2%}")
    print("\n------------------------------------------------\n")
    print("dyndcs_conv_step_num_dict:")
    dyndcs_conv_step_dict = dict(sorted(dyndcs_conv_step_dict.items()))
    total_counts = sum(dyndcs_conv_step_dict.values())
    for conv_step in dyndcs_conv_step_dict:
        counts = dyndcs_conv_step_dict[conv_step]
        propotion = counts / total_counts
        print(f"conv_step_num:{conv_step}, counts:{counts}, propotion:{propotion:.2%}")
    print("\n------------------------------------------------\n")

# prob
def test_lenet5():
    with open('/home/leitaoming/Data/FADESim_data/exp_data/lenet5_mnist/layer2_spec_xbr128_cellbit2_simple_04_04_xbr_output.pkl', 'rb') as f:
        output_dict_list_04_04 = pickle.load(f)
    with open('/home/leitaoming/Data/FADESim_data/exp_data/lenet5_mnist/layer2_spec_xbr128_cellbit2_simple_04_59_xbr_output.pkl', 'rb') as f:
        output_dict_list_04_59 = pickle.load(f)
    with open('/home/leitaoming/Data/FADESim_data/exp_data/lenet5_mnist/layer2_spec_xbr128_cellbit2_simple_59_04_xbr_output.pkl', 'rb') as f:
        output_dict_list_59_04 = pickle.load(f)
    with open('/home/leitaoming/Data/FADESim_data/exp_data/lenet5_mnist/layer2_spec_xbr128_cellbit2_simple_59_59_xbr_output.pkl', 'rb') as f:
        output_dict_list_59_59 = pickle.load(f)
    
    resolution = 8
    dcs_conv_step_num_threshold = 16

    layer_naive_conv_step_num = naive.cal_layer_cmp_times(output_dict_list_04_04, resolution)
    layer_dcs_conv_step_num_04_04, _, _ = dcs.cal_layer_cmp_times(output_dict_list_04_04, output_dict_list_04_04, resolution, dcs_conv_step_num_threshold)
    # layer_dcs_conv_step_num_04_59, _, _ = dcs.cal_layer_cmp_times(output_dict_list_04_04, output_dict_list_04_59, resolution, dcs_conv_step_num_threshold)
    # layer_dcs_conv_step_num_59_04, _, _ = dcs.cal_layer_cmp_times(output_dict_list_04_04, output_dict_list_59_04, resolution, dcs_conv_step_num_threshold)
    layer_dcs_conv_step_num_59_59, _, _ = dcs.cal_layer_cmp_times(output_dict_list_04_04, output_dict_list_59_59, resolution, dcs_conv_step_num_threshold)

    print(f"dcs_04_04 to naive:{layer_dcs_conv_step_num_04_04 / layer_naive_conv_step_num : .2%}")
    # print(f"dcs_04_59 to naive:{layer_dcs_conv_step_num_04_59 / layer_naive_conv_step_num : .2%}")
    # print(f"dcs_59_04 to naive:{layer_dcs_conv_step_num_59_04 / layer_naive_conv_step_num : .2%}")
    print(f"dcs_59_59 to naive:{layer_dcs_conv_step_num_59_59 / layer_naive_conv_step_num : .2%}")


# prob
def test_vgg11_train():
    resolution = 8
    dcs_conv_step_num_threshold = 16
    data_path, layer_num, epoch_num = '/home/leitaoming/Data/FADESim_data/exp_data/train/vgg11_cifar10/', 11, 10
    
    for layer_no in range(layer_num):
        test_file_0 = '/layer' + str(layer_no) + '_spec_xbr128_cellbit2_simp_epoch0_xbr_output.pkl'
        test_file_9 = '/layer' + str(layer_no) + '_spec_xbr128_cellbit2_simp_epoch9_xbr_output.pkl'

        try:
            with open(data_path+test_file_0, 'rb') as f:
                dict_list_0 = pickle.load(f)
            with open(data_path+test_file_9, 'rb') as f:
                dict_list_9 = pickle.load(f)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Unexcept error:{e}")

        layer_naive_conv_step_num_0 = naive.cal_layer_cmp_times(dict_list_0, resolution)
        layer_naive_conv_step_num_9 = naive.cal_layer_cmp_times(dict_list_9, resolution)
        
        layer_dcs_conv_step_num_0, _, _ = dcs.cal_layer_cmp_times(dict_list_0, dict_list_0, resolution, dcs_conv_step_num_threshold)
        layer_dcs_conv_step_num_9, _, _ = dcs.cal_layer_cmp_times(dict_list_0, dict_list_9, resolution, dcs_conv_step_num_threshold)

        print(f"layer{layer_no}")
        print(layer_naive_conv_step_num_0, layer_naive_conv_step_num_9)
        print(f"dcs_0 to naive:{layer_dcs_conv_step_num_0 / layer_naive_conv_step_num_0 : .2%}")
        print(f"dcs_9 to naive:{layer_dcs_conv_step_num_9 / layer_naive_conv_step_num_9 : .2%}")


if __name__ == "__main__":
    # test_convert()
    # test_dyndcs()
    # test_dyndcs_xbr()
    # test_dyndcs_full()
    # nn_test_final()
    # read_input_dict_list()
    # read_cycle_list()
    # read_dict_list()
    read_train_dict_list()
    # test_lenet5()
    # test_vgg11_train()
