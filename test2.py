import torch

# 假设 a 是一个已有的 Tensor
a = torch.tensor([[1, -2, 3], [-4, 5, -6]])

# 使用 torch.clamp 将负数变为 0
a_clamped = - torch.clamp(a, max=0)

print(a_clamped)
