import pickle
from src.adc_algorithm import dcs, naive, tailor, trq, adc_energy_model


# alexnet
# vgg11
# vgg16
# resnet18
def nn_test_final():
    data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/alexnet_cifar10/', 7
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/vgg11_cifar10/', 11
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/vgg16_imagenet/', 16
    # data_path, layer_num = '/home/leitaoming/Data/FADESim_data/exp_data/resnet18_imagenet/', 21

    resolution = 8
    spec_cmp_times_threshold = 10

    total_spec_cmp_times = 0
    total_naive_cmp_times = 0
    total_trq_cmp_times = 0

    spec_cmp_times_dict = {}
    # spec_para_list_dict = {}
    # trq_para_list_dict = {}

    for layer_no in range(layer_num):
        sample_file = '/layer' + str(layer_no) + '_spec_xbr128_cellbit2_sample_xbr_output.pkl' 
        test_file = '/layer' + str(layer_no) + '_spec_xbr128_cellbit2_test_xbr_output.pkl'
        trq_sample_file = '/layer' + str(layer_no) + '_trq_xbr128_cellbit2_test_xbr_output.pkl' 
        # trq_test_file = '/layer' + str(layer_no) + '_trq_xbr128_cellbit2_test_xbr_output.pkl'

        try:
            with open(data_path+sample_file, 'rb') as f:
                output_dict_list_sample = pickle.load(f)
            with open(data_path+trq_sample_file, 'rb') as f:
                trq_output_dict_list_sample = pickle.load(f)
            with open(data_path+test_file, 'rb') as f:
                output_dict_list_test = pickle.load(f)
            # with open(data_path+trq_test_file, 'rb') as f:
            #     trq_output_dict_list_test = pickle.load(f)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Unexcept error:{e}")

        layer_spec_cmp_times, layer_spec_cmp_times_dict, _ = dcs.cal_layer_cmp_times(
            output_dict_list_sample, output_dict_list_test, resolution, spec_cmp_times_threshold)
        layer_naive_cmp_times = naive.cal_layer_cmp_times(output_dict_list_test, resolution)
        layer_trq_cmp_times, _ = trq.cal_layer_cmp_times(trq_output_dict_list_sample, output_dict_list_test, resolution)

        spec_cmp_times_dict = dcs.merge_dict(spec_cmp_times_dict, layer_spec_cmp_times_dict)

        total_spec_cmp_times += layer_spec_cmp_times
        total_naive_cmp_times += layer_naive_cmp_times
        total_trq_cmp_times += layer_trq_cmp_times

        print(f"\nlayer_no={layer_no}")
        print(f"spec_cmp_times:{layer_spec_cmp_times}")
        print(f"naive_cmp_times:{layer_naive_cmp_times}")
        print(f"trq_cmp_times:{layer_trq_cmp_times}")
        print(f"spec to naive:{layer_spec_cmp_times / layer_naive_cmp_times : .2%}")
        print(f"trq to naive:{layer_trq_cmp_times / layer_naive_cmp_times : .2%}")
        print(f"spec to trq:{layer_spec_cmp_times / layer_trq_cmp_times : .2%}")

    print("\n------------------------------------------------\n")
    print(f"total_spec_cmp_times:{total_spec_cmp_times}")
    print(f"total_naive_cmp_times:{total_naive_cmp_times}")
    print(f"total_trq_cmp_times:{total_trq_cmp_times}")
    print(f"total spec to naive:{total_spec_cmp_times / total_naive_cmp_times : .2%}")
    print(f"total trq to naive:{total_trq_cmp_times / total_naive_cmp_times : .2%}")
    print(f"total spec to trq:{total_spec_cmp_times / total_trq_cmp_times : .2%}")
    print("\n------------------------------------------------\n")
    print("spec_cmp_times_dict:")
    spec_cmp_times_dict = dict(sorted(spec_cmp_times_dict.items()))
    total_counts = sum(spec_cmp_times_dict.values())
    for cmp_times in spec_cmp_times_dict:
        counts = spec_cmp_times_dict[cmp_times]
        propotion = counts / total_counts
        print(f"cmp_times:{cmp_times}, counts:{counts}, propotion:{propotion:.2%}")
    print("\n------------------------------------------------\n")

    energy_model = adc_energy_model.SAR_ADC_Energy(resolution)
    spec_energy_norm = energy_model.cal_adc_energy_norm(spec_cmp_times_dict)
    print(f"normalized spec adc energy:{spec_energy_norm : .2%}")

if __name__ == '__main__':
    nn_test_final()
