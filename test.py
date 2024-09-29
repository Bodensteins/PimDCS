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
    ], dtype=torch.float)

    test_input = torch.tensor([
        [1, 1, 2, -2, 1],
        [1, -1, 2, 2, 0],
    ], dtype=torch.float)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    print(f"device: {device}")

    test_matrix = test_matrix.to(device)
    test_input = test_input.to(device)

    weigth_bit = 3
    input_bit = 3

    mat_mul_manager = mmm.FixedPointMatMulManager(test_matrix, weigth_bit, input_bit)
    mat_mul_manager_ou = mmm.FixedPointMatMulManager_OU(test_matrix, weigth_bit, input_bit, ou_size=(2, 4))
    ou_data_analyzer = fpDA.FpDataAnalyzer(mat_mul_manager_ou)
    
    real_output = torch.matmul(test_input, test_matrix)
    test_output = mat_mul_manager.mat_mul(test_input)
    test_output_ou = mat_mul_manager_ou.mat_mul(test_input)

    print("real:")
    print(real_output)
    print("test:")
    print(test_output)
    print("ou test:")
    print(test_output_ou)

    print(mat_mul_manager_ou.fp_weight_slices)
    print(mat_mul_manager_ou.fp_input_slices)
    print(mat_mul_manager_ou.fp_input_split)

    ou_data_analyzer.update_input_sparsity()
    ou_data_analyzer.update_weight_sparsity()
    ou_data_analyzer.update_bit_vec_counts()
    ou_data_analyzer.update_bit_vec_ones_num()

    ou_data_analyzer.print_statistic()


if __name__ == '__main__':
    main()