import torch
import pickle
from src.pimtorch.config.globalCfg import globalCfg as cfg , TensorType
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.fixedPointDataAnalyze as fpDA
import src.pimtorch.nn.matMulManager as mmm
import src.pimtorch.nn.matMulManager_trq as mmm_trq
import src.pimtorch.nn.matMulManager_spec as mmm_spec
import src.pimtorch.nn.matMulManager_tailor as mmm_tailor


def main():
    # test_matrix = torch.rand(size=(10, 12)) * 64 - 32

    # test_input = torch.tensor([
    #     [10, 11, 12, 13, 14, -9, 8, 7, 63, -10],
    #     [18, -21, 42, 23, 4, 19, 9, 27, -23, 22],
    #     [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # ], dtype=torch.float)

    test_matrix = torch.tensor([
        [2, 1, 2, -2, 1, 2],
        [1, 2, 2, 1, 2, -1],
        [1, 2, -1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
        [1, -1, -1, -1, -1, -1],
        [2, 2, 2, 2, 2, 2],
    ], dtype=torch.float)

    test_input = torch.tensor([
        [1.23, 1, 2, 0.2, -1, 0, 1.44, 2],
        [1, -1.33, 2, -2, 0, 2, 2, 1],
    ], dtype=torch.float)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    print(f"device: {device}")

    test_matrix = test_matrix.to(device)
    test_input = test_input.to(device)

    weight_bit = 3
    input_bit = 8

    # mat_mul_manager = mmm.FixedPointMatMulManager(test_matrix, weight_bit, input_bit)
    # mat_mul_manager_ou = mmm.FixedPointMatMulManager_OU(test_matrix, weight_bit, input_bit, ou_size=(4, 4))
    # mat_mul_manager_pn = mmm.FixedPointMatMulManager_OU_PN(test_matrix, weight_bit, input_bit, ou_size=(4, 4))
    mat_mul_manager_ss = mmm_trq.FixedPointMatMulManager_TRQ(test_matrix, weight_bit, input_bit, ou_size=(4, 4))
    # mat_mul_manager_os = mmm.FixedPointMatMulManager_OuSample(test_matrix, weight_bit, input_bit, ou_size=(4, 4), ou_addr=(0, 0))
    mat_mul_manager_spec = mmm_spec.FixedPointMatMulManager_Spec(test_matrix, weight_bit, input_bit, ou_size=(4, 4), weight_slice_bit=1)
    mat_mul_manager_trq = mmm_trq.FixedPointMatMulManager_TRQ(test_matrix, weight_bit, input_bit, ou_size=(4, 4), weight_slice_bit=1)

    # ou_data_analyzer = fpDA.FpDataAnalyzer(mat_mul_manager_ou)
    # pn_data_analyzer = fpDA.FpDataAnalyzer_PN(mat_mul_manager_pn)
    # os_data_analyzer = fpDA.FpDataAnalyzer_OuSample(mat_mul_manager_os)
    spec_data_analyzer = fpDA.FpDataAnalyzer_Spec(mat_mul_manager_spec)
    trq_data_analyzer = fpDA.FpDataAnalyzer_TRQ(mat_mul_manager_trq)
    # print(mat_mul_manager_pn.weight_s)

    mat_mul_manager_spec.set_data_analyzer(spec_data_analyzer)
    mat_mul_manager_trq.set_data_analyzer(trq_data_analyzer)

    # print(mat_mul_manager_os.fp_weight_sample_slice_pos)
    # print(mat_mul_manager_os.fp_weight_sample_slice_neg)

    real_output = torch.matmul(test_input, test_matrix)
    # test_output_ou = mat_mul_manager_ou.mat_mul(test_input)
    test_output_spec = mat_mul_manager_spec.mat_mul(test_input)
    # test_output_ss_f = mat_mul_manager_ss.fake_mat_mul(test_input)
    # test_output_pn = mat_mul_manager_pn.fake_mat_mul(test_input)
    # test_output_os = mat_mul_manager_os.fake_mat_mul(test_input)
    test_output_trq = mat_mul_manager_trq.mat_mul(test_input)


    print("real:")
    print(real_output)
    # print("os fake test:")
    # print(test_output_os)
    print("trq test:")
    print(test_output_trq)
    # print("ou test:")
    # print(test_output_ou)
    print("spec test:")
    print(test_output_spec)
    # print("ss fake test:")
    # print(test_output_ss_f)

    # print(mat_mul_manager_os.fp_ou_column_output_pp)
    # os_data_analyzer.update_ou_column_output_num()
    # os_data_analyzer.print_statistic()
    # spec_data_analyzer.save_statistic('/home/leitaoming/Data/FADESim_data/exp_data/test')
    # with open('/home/leitaoming/Data/FADESim_data/exp_data/test/layer0_test_xbr_output.pkl', 'rb') as f:
    #     output_dict_list_test = pickle.load(f)
    # print(output_dict_list_test[0][0][0][0][0])
    # print(output_dict_list_test[1][1][0][0][0])
    # print(output_dict_list_test[0][7][1][0][0])


def trq_test():
    test_matrix = torch.tensor([
        [12, 33, 2, 5, 10, 8, 4, 10],
        [1, 15, 16, 17, 7, 9, 10, 8],
    ], dtype=torch.int8)

    out_slice = mmm_trq.FixedPointMatMulManager_TRQ.twin_range_quantize(test_matrix, 5, 3, 2)
    print(out_slice)
    # print(torch.min(out_slice)>=0)


def tailor_test():
    test_matrix = torch.tensor([
        [12, 33, 2, 5, 10, 8, 4, 10],
        [1, 15, 16, 17, 7, 9, 10, 8],
    ], dtype=torch.int8)
    mm_manager = mmm_tailor.FixedPointMatMulManager_Tailor(test_matrix, 4, 4, (4, 4), (4, 4), adc_resolution=5, n_noise_max=12)
    out_slice = mm_manager.tailor(test_matrix, 1, 1)
    print(out_slice)

if __name__ == '__main__':
    main()
    # trq_test()
    # tailor_test()
