import torch
from src.pimtorch.config.globalCfg import globalCfg as cfg , TensorType
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.fixedPointDataAnalyze as fpDA
import math


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    print(f"device: {device}")

    # 假设 b 的形状为 [ou_row_num, bsize, ou_col_num]
    ou_row_num = 3  # 示例值
    bsize = 4  # 示例值
    ou_col_num = 2  # 示例值

    # 随机生成 Tensor b
    b = torch.randint(0, 2, (ou_row_num, bsize, ou_col_num), device=device)  # 生成随机整数 Tensor

    # 打印 Tensor b
    print("Tensor b:")
    print(b)

    # 用于保存每个二维子 Tensor 中子向量出现次数的列表
    result_list = []

    # 遍历每个二维子 Tensor c[bsize, ou_col_num]
    for c in b:
        # 将每个子向量转换为 tuple，并将其作为字典的 key
        # 使用 torch.unique 对每个子向量进行编码
        unique, counts = torch.unique(c, return_counts=True, dim=0)

        # 将 unique 中的每个子向量转换为 tuple
        unique_tuples = [tuple(row.tolist()) for row in unique]

        # 创建一个字典，键为 tensor，值为其出现的次数
        vector_count = dict(zip(unique_tuples, counts.tolist()))

        # 将统计结果字典添加到结果列表中
        result_list.append(vector_count)

    # 打印每个 c 的子向量出现次数
    for idx, count_dict in enumerate(result_list):
        print(f"二维子 Tensor c[{idx}] 中子向量出现次数统计:")
        print(count_dict)


if __name__ == '__main__':
    main()