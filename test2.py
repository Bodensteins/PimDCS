import torch
from src.pimtorch.config.globalCfg import globalCfg as cfg , TensorType
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.fixedPointDataAnalyze as fpDA
import math


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")
    print(f"device: {device}")

    # 假设 a 是一个形状为 [h, w*b] 的 Tensor
    h = 6   # 示例值
    w = 4   # 示例值
    b = 3   # 示例值
    a = torch.arange(h * w * b).view(h, w * b)  # 创建一个示例 Tensor，形状为 [h, w*b]

    # 打印原始 Tensor
    print("原始 Tensor a:")
    print(a)

    # 第一步：将 Tensor a 的形状调整为 [h, b, w]
    # 将原始 Tensor a 重塑为 [h, b, w]，-1 自动计算宽度 w
    reshaped_a = a.view(h, b, w)

    # 第二步：调整维度顺序为 [b, h, w]
    c = reshaped_a.permute(1, 0, 2)  # 使用 permute 将维度顺序从 [h, b, w] 调整为 [b, h, w]

    # 打印结果 Tensor c
    print("新的三维 Tensor c 的形状:")
    print(c.shape)
    print(c)



if __name__ == '__main__':
    main()