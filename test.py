import torch
from src.pimtorch.config.globalCfg import globalCfg as cfg , TensorType
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.fixedPointDataAnalyze as fpDA
import src.pimtorch.nn.matMulManager as mmm


def main():
    # test_matrix = torch.rand(size=(10, 12)) * 64 - 32

    # test_input = torch.tensor([
    #     [10, 11, 12, 13, 14, -9, 8, 7, 63, -10],
    #     [18, -21, 42, 23, 4, 19, 9, 27, -23, 22],
    #     [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # ], dtype=torch.float)

    test_matrix = torch.tensor([
        [-2, 1, 2, -2, 1, 2],
        [1, 2, 2, 1, 2, -1],
        [1, 2, -1, 2, 1, 2],
        [-1, 2, 1, 2, 1, 2],
        [-1, 2, 1, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1],
        [2, 2, 2, 2, 2, 2],
    ], dtype=torch.float)

    test_input = torch.tensor([
        [1, 1, 2, -2, 1, 0, 1, -1],
        [1, -1, 2, 2, 0, -2, -2, 1],
    ], dtype=torch.float)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    print(f"device: {device}")

    test_matrix = test_matrix.to(device)
    test_input = test_input.to(device)

    weigth_bit = 2
    input_bit = 3

    mat_mul_manager = mmm.FixedPointMatMulManager(test_matrix, weigth_bit, input_bit)
    mat_mul_manager_ou = mmm.FixedPointMatMulManager_OU(test_matrix, weigth_bit, input_bit, ou_size=(4, 4))
    mat_mul_manager_pn = mmm.FixedPointMatMulManager_OU_PN(test_matrix, weigth_bit, input_bit, ou_size=(4, 4))
    ou_data_analyzer = fpDA.FpDataAnalyzer(mat_mul_manager_ou)
    pn_data_analyzer = fpDA.FpDataAnalyzer_PN(mat_mul_manager_pn)
    # print(mat_mul_manager_pn.weight_s)

    real_output = torch.matmul(test_input, test_matrix)
    test_output = mat_mul_manager.mat_mul(test_input)
    test_output_pn = mat_mul_manager_pn.mat_mul(test_input)

    # print("real:")
    # print(real_output)
    # print("test:")
    # print(test_output)
    # print("ou test:")
    # print(test_output_pn)

    # print(mat_mul_manager_pn.fp_weight_slices)
    # print(mat_mul_manager_pn.fp_input_slices)
    # print(mat_mul_manager_pn.fp_input_split)
    # print(mat_mul_manager_pn.fp_weight_split_pos)
    # print(mat_mul_manager_pn.fp_weight_split_neg)

    print(mat_mul_manager_pn.fp_weight_slices_pos)
    print(mat_mul_manager_pn.fp_weight_slices_neg)

    pn_data_analyzer.update_all_weight_statistic()
    pn_data_analyzer.update_all_input_statistic()

    pn_data_analyzer.print_statistic()

    # print(mat_mul_manager_pn.fp_weight_pos)
    # print(mat_mul_manager_pn.fp_weight_neg)


if __name__ == '__main__':
    main()