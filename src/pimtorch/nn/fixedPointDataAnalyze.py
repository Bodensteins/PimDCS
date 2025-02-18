import math
import torch
from torch import Tensor
from src.pimtorch.config.globalCfg import TensorType, globalCfg
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.matMulManager as mmm
from concurrent.futures import ThreadPoolExecutor


# 分析输入和权重数据特征
# 神经网络的每一层对应一个Analyzer
# 已测试
class FpDataAnalyzer:
    def __init__(self, mm_manager:mmm.FixedPointMatMulManager_OU=None) -> None:
        self.mm_manager = mm_manager
        # 目前只考虑bit_width=1的情况
        assert mm_manager.weight_slice_bit == 1 and mm_manager.input_slice_bit == 1
        assert mm_manager != None

        # 统计权重比特分片稀疏度（零元占比），list[bit]       
        self.weight_bit_slices_sparsity = []
        # 统计输入比特分片稀疏度（零元占比），list[bit]
        self.input_bit_slices_sparsity = []
        # 统计OU粒度大小输入向量的出现频率，list[ou_row_num]，每行OU对应一个频率dict，dict的key为OU输入向量，value为该向量出现的次数
        self.input_bit_vec_counts_dict_list = []
        # 统计OU粒度大小中不同1比特数量的输入向量的出现频率，dict
        self.input_bit_vec_ones_num_dict = {}

        for _ in range(self.mm_manager.weight_bit_width):
            self.weight_bit_slices_sparsity.append(0)  # 零元比例
        for _ in range(self.mm_manager.input_bit_width):
            self.input_bit_slices_sparsity.append(0)   # 零元数量
        for _ in range(math.ceil(self.mm_manager.in_size / self.mm_manager.ou_size[0])):
            self.input_bit_vec_counts_dict_list.append({})

        self.total_input_slice_bit_num = 0

    # 统计权重矩阵每个比特分片的稀疏度
    def update_weight_sparsity(self) -> None:
        assert self.mm_manager.fp_weight_slices != None
        # 若各比特分片合在了一起，则需重新拆开
        weight_slices = self.mm_manager.fp_weight_slices
        if self.mm_manager.weight_slice_cat_dim == 0:
            split_slices = torch.split(self.mm_manager.fp_weight_slices, self.mm_manager.fp_weight.shape[1], dim=1)
            weight_slices = torch.stack(split_slices, dim=0)
        # 总元素数量
        total_elements_num = weight_slices.shape[1] * weight_slices.shape[2]
        # 零元数量
        zero_count = (weight_slices == 0).sum(dim=(1, 2))
        # 零元比例
        zero_ratio = (zero_count / total_elements_num)
        # 更新权重分片稀疏度列表
        for i in range(self.mm_manager.weight_bit_width):
            self.weight_bit_slices_sparsity[i] = zero_ratio[i].item()

    # 统计输入向量每个比特分片的稀疏度
    def update_input_sparsity(self) -> None:
        assert self.mm_manager.fp_input_slices != None
        # 若各比特分片合在了一起，则需重新拆开
        input_slices = self.mm_manager.fp_input_slices
        if self.mm_manager.input_slice_cat_dim == 1:
            input_slices = self.mm_manager.fp_input_slices.reshape(
                self.mm_manager.input_bit_width, self.mm_manager.fp_input.shape[0], self.mm_manager.fp_input.shape[1])
        # 总元素数量
        total_elements_num = input_slices.shape[1] * input_slices.shape[2]
        self.total_input_slice_bit_num += total_elements_num
        # 零元数量
        zero_count = (input_slices == 0).sum(dim=(1, 2))
        # 更新输入分片稀疏度列表
        for i in range(self.mm_manager.input_bit_width):
            self.input_bit_slices_sparsity[i] += zero_count[i].item()

    # 统计OU粒度大小输入向量的出现频率
    # 是否应该让unique和counts一直留在cuda上？
    def update_input_bit_vec_counts(self) -> None:
        assert self.mm_manager.fp_input_split != None
        input_split = self.mm_manager.fp_input_split
        # input_split维度大小分别为[ou_row_num, batch_size*bit_width, ou_row_size]
        for i, input_block in enumerate(input_split):
            # 使用torch.unique 对每个子向量进行次数统计
            unique, counts = torch.unique(input_block, return_counts=True, dim=0)
            unique_cpu = unique.cpu()
            # 利用线程池进行并行转换
            with ThreadPoolExecutor() as executor:
                # unique_keys = list(executor.map(FpDataAnalyzer.binary_tensor_to_int, unique_cpu)) # 转换为整数
                unique_keys = list(executor.map(FpDataAnalyzer.tensor_to_tuple, unique_cpu)) # 转换为tuple
            # 创建一个字典，键为tensor的key，值为其出现的次数
            vector_counts = dict(zip(unique_keys, counts.tolist()))
            # 更新bit_vec_counts字典
            self.input_bit_vec_counts_dict_list[i].update(vector_counts)

    # tensor映射为tuple
    @staticmethod
    def tensor_to_tuple(tensor:Tensor):
        return tuple(tensor.tolist())
    
    # tensor元素只有0和1则映射为整数
    @staticmethod
    def binary_tensor_to_int(tensor:Tensor):
        return int("".join(map(str, tensor.tolist())), 2)

    # 统计OU粒度大小中不同1比特数量的输入向量的出现频率
    def update_input_bit_vec_ones_num(self) -> None:
        assert self.mm_manager.fp_input_split != None
        input_split = self.mm_manager.fp_input_split
        # input_split维度大小分别为[ou_row_num, batch_size*bit_width, ou_row_size]
        # 因为元素只有0和1，所以先沿着最内层维度求和，即计算每个子向量中的'1'数量，然后展平为一维
        ones_num_counts = input_split.sum(dim=2).flatten()
        # 使用torch.unique统计不同'1'数量的出现频率
        ones_num, counts = torch.unique(ones_num_counts, return_counts=True)
        # 更新bit_vec_ones_num_counts字典
        for n, c in zip(ones_num, counts):
            if self.input_bit_vec_ones_num_dict.__contains__(n.item()):
                self.input_bit_vec_ones_num_dict[n.item()] += c.item()
            else:
                self.input_bit_vec_ones_num_dict[n.item()] = c.item()

    # 重置输入记录
    def reset_input_statistic(self):    
        self.input_bit_slices_sparsity = []
        self.input_bit_vec_counts_dict_list = []
        self.input_bit_vec_ones_num_dict = {}
        
        for _ in range(self.mm_manager.input_bit_width):
            self.input_bit_slices_sparsity.append(0)
        for _ in range(math.ceil(self.mm_manager.in_size / self.mm_manager.ou_size[0])):
            self.input_bit_vec_counts_dict_list.append({})

        self.total_input_slice_bit_num = 0

    # 更新所有输入数据特征统计
    def update_all_input_statistic(self):
        self.update_input_sparsity()
        # self.update_input_bit_vec_counts()
        self.update_input_bit_vec_ones_num()

    # 更新所有权重数据特征统计
    def update_all_weight_statistic(self):
        self.update_weight_sparsity()

    # 打印结果
    def print_statistic(self):
        print("权重比特分片稀疏度:")
        for i, sparsity in enumerate(self.weight_bit_slices_sparsity):
            print(f"bit {i}, sparsity:{sparsity:.2%}")
        print("\n输入比特分片稀疏度:")
        for i, num in enumerate(self.input_bit_slices_sparsity):
            sparsity = num / self.total_input_slice_bit_num 
            print(f"bit {i}, sparsity:{sparsity:.2%}")
        # print("\nOU粒度输入向量比特出现次数和频率:")
        # for i, counts_dict in enumerate(self.input_bit_vec_counts_dict_list):
        #     total_counts = sum(counts_dict.values())
        #     print(f"ou row {i}:")
        #     sorted_counts = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        #     for bit_vec, counts in sorted_counts:
        #         propotion = counts / total_counts
        #         print(f"bit vec:{bit_vec}, counts:{counts}, propotion:{propotion:.2%}")
        print("\n不同1数量的输入向量的出现次数和频率:")
        total_counts = sum(self.input_bit_vec_ones_num_dict.values())
        print(self.input_bit_vec_ones_num_dict)
        for ones_num in self.input_bit_vec_ones_num_dict:
            counts = self.input_bit_vec_ones_num_dict[ones_num]
            propotion = counts / total_counts
            if ones_num != 0:
                non_zero_propotion = counts / (total_counts - self.input_bit_vec_ones_num_dict[0])
            else:
                non_zero_propotion = 0
            print(f"ones num:{ones_num}, counts:{counts}, propotion:{propotion:.2%}, non zero propotion:{non_zero_propotion:.2%}")

    def save_statistic(self):
        pass


# 权重分为正负矩阵
class FpDataAnalyzer_PN(FpDataAnalyzer):
    def __init__(self, mm_manager:mmm.FixedPointMatMulManager_OU_PN=None) -> None:
        super().__init__(mm_manager)
        self.mm_manager = mm_manager

        # 统计权重比特分片稀疏度（零元占比），分正负矩阵，list[bit]       
        self.weight_bit_slices_pos_sparsity = []
        self.weight_bit_slices_neg_sparsity = []
        for _ in range(self.mm_manager.weight_bit_width):
            self.weight_bit_slices_pos_sparsity.append(0)  # 零元比例
            self.weight_bit_slices_neg_sparsity.append(0)

        # 统计OU粒度大小中不同1比特数量的权重向量的出现频率，分正负阵列，dict
        self.weight_bit_vec_ones_num_dict_pos = {}
        self.weight_bit_vec_ones_num_dict_neg = {}

        # 统计不同OU输出结果的出现频率
        self.ou_column_output_num_dict = {}



    def update_weight_sparsity(self) -> None:
        assert self.mm_manager != None
        # 若各比特分片合在了一起，则需重新拆开
        weight_slices_pos = self.mm_manager.fp_weight_slices_pos
        weight_slices_neg = self.mm_manager.fp_weight_slices_neg
        if self.mm_manager.weight_slice_cat_dim == 0:
            split_slices_pos = torch.split(self.mm_manager.fp_weight_slices_pos, self.mm_manager.fp_weight_pos.shape[1], dim=1)
            weight_slices_pos = torch.stack(split_slices_pos, dim=0)
            split_slices_neg = torch.split(self.mm_manager.fp_weight_slices_neg, self.mm_manager.fp_weight_neg.shape[1], dim=1)
            weight_slices_neg = torch.stack(split_slices_neg, dim=0)
            
        # 总元素数量
        total_elements_num = weight_slices_pos.shape[1] * weight_slices_pos.shape[2]
        # 零元数量
        zero_count_pos = (weight_slices_pos == 0).sum(dim=(1, 2))
        zero_count_neg = (weight_slices_neg == 0).sum(dim=(1, 2))
        # 零元比例
        zero_ratio_pos = (zero_count_pos / total_elements_num)
        zero_ratio_neg = (zero_count_neg / total_elements_num)
        # 更新权重分片稀疏度列表
        for i in range(self.mm_manager.weight_bit_width):
            self.weight_bit_slices_pos_sparsity[i] = zero_ratio_pos[i].item()
            self.weight_bit_slices_neg_sparsity[i] = zero_ratio_neg[i].item()

    def update_weight_bit_vec_ones_num(self) -> None:
        assert self.mm_manager.fp_weight_split_pos != None
        assert self.mm_manager.fp_weight_split_neg != None
        weight_split_pos = self.mm_manager.fp_weight_split_pos
        weight_split_neg = self.mm_manager.fp_weight_split_neg
        # weight_split维度大小为[ou_row_num, ou_col_num * bit_width, ou_row_size, ou_col_size]
        # 因为元素只有0和1，所以先沿第2维度求和，即计算每个OU列中的'1'数量，然后展平为一维
        ones_num_counts_pos = weight_split_pos.sum(dim=2).flatten()
        ones_num_counts_neg = weight_split_neg.sum(dim=2).flatten()
        # 使用torch.unique统计不同'1'数量的出现频率
        ones_num_pos, counts_pos = torch.unique(ones_num_counts_pos, return_counts=True)
        ones_num_neg, counts_neg = torch.unique(ones_num_counts_neg, return_counts=True)
        # 更新bit_vec_ones_num_counts字典
        for n_pos, c_pos in zip(ones_num_pos, counts_pos):
            if self.weight_bit_vec_ones_num_dict_pos.__contains__(n_pos.item()):
                self.weight_bit_vec_ones_num_dict_pos[n_pos.item()] += c_pos.item()
            else:
                self.weight_bit_vec_ones_num_dict_pos[n_pos.item()] = c_pos.item()
        for n_neg, c_neg in zip(ones_num_neg, counts_neg):
            if self.weight_bit_vec_ones_num_dict_neg.__contains__(n_neg.item()):
                self.weight_bit_vec_ones_num_dict_neg[n_neg.item()] += c_neg.item()
            else:
                self.weight_bit_vec_ones_num_dict_neg[n_neg.item()] = c_neg.item()

    def update_ou_column_output_num(self) -> None:
        assert self.mm_manager.fp_ou_column_output != None
        # device = torch.device("cpu")
        ou_column_output = self.mm_manager.fp_ou_column_output#.to(torch.device("cpu"))
        out_value, counts = torch.unique(ou_column_output, return_counts=True)
        for n, c in zip(out_value, counts):
            if self.ou_column_output_num_dict.__contains__(n.item()):
                self.ou_column_output_num_dict[n.item()] += c.item()
            else:
                self.ou_column_output_num_dict[n.item()] = c.item()

    def update_all_weight_statistic(self):
        self.update_weight_sparsity()
        # self.update_weight_bit_vec_ones_num()
        # self.update_ou_column_output_num()

    def print_statistic(self):
        print("权重比特分片稀疏度:")
        for i, sparsity in enumerate(self.weight_bit_slices_pos_sparsity):
            print(f"bit {i}, positive sparsity:{sparsity:.2%}, negative sparsity:{self.weight_bit_slices_neg_sparsity[i]:.2%}")

        print("\n输入比特分片稀疏度:")
        for i, num in enumerate(self.input_bit_slices_sparsity):
            sparsity = num / self.total_input_slice_bit_num 
            print(f"bit {i}, sparsity:{sparsity:.2%}")

        # print("\nOU粒度输入向量比特出现次数和频率:")
        # for i, counts_dict in enumerate(self.input_bit_vec_counts_dict_list):
        #     total_counts = sum(counts_dict.values())
        #     print(f"ou row {i}:")
        #     sorted_counts = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        #     for bit_vec, counts in sorted_counts:
        #         propotion = counts / total_counts
        #         print(f"bit vec:{bit_vec}, counts:{counts}, propotion:{propotion:.2%}")
            
        # print("\n不同1数量的权重OU列的出现次数和频率:")
        # total_counts_pos = sum(self.weight_bit_vec_ones_num_dict_pos.values())
        # total_counts_neg = sum(self.weight_bit_vec_ones_num_dict_neg.values())
        # for ones_num_pos in self.weight_bit_vec_ones_num_dict_pos:
        #     counts_pos = self.weight_bit_vec_ones_num_dict_pos[ones_num_pos]
        #     propotion_pos = counts_pos / total_counts_pos
        #     print(f"positive ou -- ones num:{ones_num_pos}, counts:{counts_pos}, propotion:{propotion_pos:.2%}")
        # for ones_num_neg in self.weight_bit_vec_ones_num_dict_neg:
        #     counts_neg = self.weight_bit_vec_ones_num_dict_neg[ones_num_neg]
        #     propotion_neg = counts_neg / total_counts_neg
        #     print(f"negative ou -- ones num:{ones_num_neg}, counts:{counts_neg}, propotion:{propotion_neg:.2%}")
        print("\n不同OU输出结果的出现次数和频率(正负OU结果相减):")
        total_counts = sum(self.ou_column_output_num_dict.values())
        for out_value in self.ou_column_output_num_dict:
            counts = self.ou_column_output_num_dict[out_value]
            propotion = counts / total_counts
            if out_value != 0:
                non_zero_propotion = counts / (total_counts - self.input_bit_vec_ones_num_dict[0])
            else:
                non_zero_propotion = 0
            print(f"ou out value:{out_value}, counts:{counts}, propotion:{propotion:.2%}, non zero propotion:{non_zero_propotion:.2%}")

        print("\n不同1数量的OU粒度输入向量的出现次数和频率:")
        total_counts = sum(self.input_bit_vec_ones_num_dict.values())
        print(self.input_bit_vec_ones_num_dict)
        if not self.input_bit_vec_ones_num_dict.__contains__(0):
            self.input_bit_vec_ones_num_dict[0] = 0
        for ones_num in self.input_bit_vec_ones_num_dict:
            counts = self.input_bit_vec_ones_num_dict[ones_num]
            propotion = counts / total_counts
            if ones_num != 0:
                non_zero_propotion = counts / (total_counts - self.input_bit_vec_ones_num_dict[0])
            else:
                non_zero_propotion = 0
            print(f"ones num:{ones_num}, counts:{counts}, propotion:{propotion:.2%}, non zero propotion:{non_zero_propotion:.2%}")