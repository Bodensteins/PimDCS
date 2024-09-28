import math
import torch
from torch import Tensor
from src.pimtorch.config.globalCfg import TensorType, globalCfg
import src.pimtorch.nn.fixedPointArithmetic as fpA
import src.pimtorch.nn.fixedPointDataAnalyze as fpDA


# 仅仅是将输入和权重分片，然后分片相乘，移位累加
# 已测试
class FixedPointMatMulManager:
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width:int,
                 input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                 fp_data_analyzer:fpDA.FpDataAnalyzer=None,
                 is_slice_in_init:bool=True) -> None:
        # 计算设备，cpu or cuda
        self.device = weight.device

        # 权重量化
        self.fp_weight, params = fpA.quantization_tensor(weight, weight_bit_width, TensorType.Normal)
        self.weight_s = params[0].item()    # 权重小数点位置
        self.weight_bit_width = weight_bit_width

        self.input_bit_width = input_bit_width

        # 输入向量、输出向量大小
        self.original_in_size = weight.shape[0]
        self.original_out_size = weight.shape[1]

        # 真实输入输出大小
        self.in_size = self.original_in_size
        self.out_size = self.original_out_size

        # 比特分片粒度
        self.input_slice_bit = input_slice_bit
        self.weight_slice_bit = weight_slice_bit

        self.fp_weight_slices = None

        # 权重比特分片结果的三维Tensor，[bits, in_size, out_size]
        if is_slice_in_init:
            self.fp_weight_slices = FixedPointMatMulManager.tensor_slice(
                self.fp_weight, self.weight_bit_width, self.weight_slice_bit)

        # 数据分析器
        self.fp_data_analyzer = fp_data_analyzer

    # 矩阵比特分片
    @staticmethod
    def tensor_slice(fp_tensor:Tensor, mat_bit_width:int, slice_bit:int=1) -> Tensor:
        total_slice_num = math.ceil(mat_bit_width / slice_bit)
        fp_mat_slices = torch.zeros((total_slice_num, *fp_tensor.shape), dtype=torch.int8, device=fp_tensor.device)
        # 按slice_bit进行分片
        for i in range(total_slice_num):
            fp_mat_slices[i] = (fp_tensor >> (i * slice_bit)) & ((1 << slice_bit) - 1)

        return fp_mat_slices

    # 矩阵乘法
    def mat_mul(self, input:Tensor) -> Tensor:
        assert input.shape[1] == self.in_size
        assert input.device == self.device
        batch_size = input.shape[0]

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        input_s = params[0].item()

        fp_input_slices = FixedPointMatMulManager.tensor_slice(fp_input, self.input_bit_width, self.input_slice_bit)
        output = torch.zeros(batch_size, self.out_size, device=self.device)

        # 简单起见，暂时只考虑1bit分片
        assert self.input_slice_bit == self.weight_slice_bit == 1
        # 每个分片做矩阵乘法，然后移位累加
        for i, input_slice in enumerate(fp_input_slices):
            for j, weight_slice in enumerate(self.fp_weight_slices):
                # 比特切片相乘
                output_slice = FixedPointMatMulManager.tensor_int_mul_cuda(input_slice, weight_slice)
                if (i == fp_input_slices.shape[0] - 1 and j < self.fp_weight_slices.shape[0] - 1) \
                    or (i < fp_input_slices.shape[0] - 1 and j == self.fp_weight_slices.shape[0] - 1):
                    # 输入切片有符号位和权重切片中有一个有符号位则取反
                    output_slice = -output_slice
                
                # 累加至最终结果
                output = output.add(output_slice.mul(2 ** (i + j + input_s + self.weight_s)))

        return output[:, 0:self.original_out_size]

    # 用cuda对两个int类型二维tensor进行乘法
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        if not mat0.is_cuda or not mat1.is_cuda:
            return torch.matmul(mat0, mat1)

        float_tensor0 = mat0.to(dtype=torch.float)
        float_tensor1 = mat1.to(dtype=torch.float)
        mul_result = float_tensor0.matmul(float_tensor1)

        return mul_result.to(dtype=torch.int64)


# 以OU为粒度进行乘加计算
# 尝试用cuda并行计算多个OU矩阵向量乘法
class FixedPointMatMulManager_OU(FixedPointMatMulManager):
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width,
                 xbr_size:tuple=globalCfg.crxShape, ou_size:tuple=globalCfg.ouShape, 
                 input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                 fp_data_analyzer:fpDA.FpDataAnalyzer=None,
                 is_slice_in_init:bool=True, is_split_in_init:bool=True) -> None:
        super().__init__(weight, weight_bit_width, input_bit_width, input_slice_bit, weight_slice_bit, is_slice_in_init=False)

        # xbr、OU大小
        self.xbr_size = xbr_size
        self.ou_size = ou_size

        # 根据OU大小对权重矩阵进行补全，使权重矩阵与OU大小对齐
        if self.original_in_size % self.ou_size[0] != 0:
            temp_size = self.ou_size[0] - self.original_in_size % self.ou_size[0]
            self.fp_weight = torch.cat(
                (self.fp_weight, torch.zeros(temp_size, self.out_size, dtype=self.fp_weight.dtype, device=self.device)), dim=0)
            self.in_size += temp_size
        if self.original_out_size % self.ou_size[1] != 0:
            temp_size = self.ou_size[1] - self.original_out_size % self.ou_size[1]
            self.fp_weight = torch.cat(
                (self.fp_weight, torch.zeros(self.in_size, temp_size, dtype=self.fp_weight.dtype, device=self.device)), dim=1)
            self.out_size += temp_size
        
        # 权重比特分片结果的二维Tensor，[in_size, out_size * bit_width]
        if is_slice_in_init:
            self.fp_weight_slices = FixedPointMatMulManager_OU.tensor_slice(
                self.fp_weight, self.weight_bit_width, self.weight_slice_bit, cat_dim=0)

        self.fp_weight_split = None

        # 权重比特分片后按OU大小拆分
        if is_split_in_init:
            self.fp_weight_split = FixedPointMatMulManager_OU.weight_split(self.fp_weight_slices, ou_size)

        # 数据分析器
        self.fp_data_analyzer = fp_data_analyzer

    # 矩阵比特分片
    # 与父类不同的是，提供将每个切片拼接起来的选项
    # cat_dim=-1为不拼接，0为横向拼接，1为纵向拼接
    @staticmethod
    def tensor_slice(fp_tensor:Tensor, mat_bit_width:int, slice_bit:int=1, cat_dim:int=-1) -> Tensor:
        total_slice_num = math.ceil(mat_bit_width / slice_bit)
        fp_mat_slices = torch.zeros((total_slice_num, *fp_tensor.shape), dtype=torch.int64, device=fp_tensor.device)
        # 按slice_bit进行分片
        for i in range(total_slice_num):
            fp_mat_slices[i] = (fp_tensor >> (i * slice_bit)) & ((1 << slice_bit) - 1)

        # 比特分片拼接，不能用view
        if cat_dim == 0:
            h = fp_mat_slices.shape[1]
            fp_mat_slices = fp_mat_slices.permute(1, 0, 2).reshape(h, -1)
        elif cat_dim == 1:
            w = fp_mat_slices.shape[2]
            fp_mat_slices = fp_mat_slices.reshape(-1, w)

        return fp_mat_slices

    # 二维的矩阵分片按固定大小拆分，获得四维Tensor，[ou_row_num, ou_col_num * bit_width, ou_row_size, ou_col_size]
    # 已测试，注意view方法需要tensor中的数据在内存中为连续存储
    @staticmethod
    def weight_split(weight_slices:Tensor, split_size:tuple) -> Tensor:
        height, width = weight_slices.shape

        assert height % split_size[0] == 0 and width % split_size[1] == 0
        
        split_num_row = height // split_size[0]
        split_num_col = width // split_size[1]

        weight_split = weight_slices.view(
            split_num_row, split_size[0], split_num_col, split_size[1]).permute(0, 2, 1, 3)
        
        return weight_split
    
    # 三维的输入分片按ou_row大小拆分，获得三维Tensor，[ou_row_num, batch_size*bit_width, ou_row_size]
    # 已测试，注意view方法需要tensor中的数据在内存中为连续存储
    @staticmethod
    def input_split(input_slices:Tensor, split_in_size:int) -> Tensor:
        # print(input_slices.shape)
        b_size, in_size = input_slices.shape
        # print(b_size,in_size,split_in_size)
        assert in_size % split_in_size == 0
        split_num = in_size // split_in_size

        input_split = input_slices.view(b_size, split_num, split_in_size).permute(1, 0, 2)
        return input_split
    
    # 以OU粒度进行的矩阵乘法
    # 已测试
    # einsum相比循环能提升多少性能？
    def mat_mul(self, input:Tensor) -> Tensor:
        assert input.device == self.device
        assert input.shape[1] == self.original_in_size
        batch_size = input.shape[0]
        ou_col_num = self.fp_weight_split.shape[1] // self.weight_bit_width
        ou_col_size = self.fp_weight_split.shape[3]

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        input_s = params[0].item()

        # 根据ou_row大小补全输入，使输入长度与ou_row大小对齐
        if self.original_in_size % self.ou_size[0] != 0:
            temp_size = self.ou_size[0] - self.original_in_size % self.ou_size[0]
            fp_input = torch.cat((fp_input, torch.zeros((batch_size, temp_size), dtype=fp_input.dtype, device=fp_input.device)), dim=1)

        # 输入按比特分片， [batch_szie*bit_width, in_size]
        fp_input_slices = FixedPointMatMulManager_OU.tensor_slice(fp_input, self.input_bit_width, self.input_slice_bit, cat_dim=1)

        # 按OU行大小拆分输入比特，[ou_row_num, batch_size*bit_width, ou_row_size]
        fp_input_split = FixedPointMatMulManager_OU.input_split(fp_input_slices, self.ou_size[0])

        # 每个OU并行执行矩阵乘法，利用einsum进行并行计算
        # 得到output[ou_row_num, ou_col_num*weight_bit_width, batch_size*input_bit_width, ou_col_size]
        output = FixedPointMatMulManager_OU.tensor_int_einsum_cuda('rbh,rchw->rcbw', fp_input_split, self.fp_weight_split)

        # 对output的结果进行移位累加
        # 先沿第0维累加，得到output[ou_col_num*weight_bit_width, batch_size*input_bit_width, ou_col_size]
        output = torch.sum(output, dim=0)

        # 按照权重比特位进行移位累加，得到output[ou_col_num, batch_size*input_bit_width, ou_col_size]
        # 这里暂未找到方法消除循环，但应该是能够并行的。涉及到对符号的判断，若要优化可能需要写CUDA核函数
        temp_out = torch.zeros((ou_col_num, batch_size*self.input_bit_width, ou_col_size), dtype=torch.int64, device=fp_input.device)
        for i in range(self.weight_bit_width):
            if i < self.weight_bit_width - 1:
                temp_out += (output[i*ou_col_num:(i+1)*ou_col_num, :, :] << i)
            else:
                temp_out -= (output[i*ou_col_num:(i+1)*ou_col_num, :, :] << i)
        output = temp_out

        # 拼接矩阵的第0维和第2维，合并ou_col_num和ou_col_size，得到output[batch_size*input_bit_width, out_size]
        output = output.permute(1, 0, 2).reshape(-1, self.out_size)

        # 按照输出比特位进行移位累加，得到output[batch_size, ou_col_size]
        # 与权重比特移位累加一样，暂未找到方法消除循环，但应该是能够并行的。涉及到对符号的判断，若要优化可能需要写CUDA核函数
        temp_out = torch.zeros((batch_size, self.out_size), dtype=torch.int64, device=fp_input.device)
        for i in range(self.input_bit_width):
            if i < self.weight_bit_width - 1:
                temp_out += (output[i*batch_size:(i+1)*batch_size, :] << i)
            else:
                temp_out -= (output[i*batch_size:(i+1)*batch_size, :] << i)
        output = temp_out.to(torch.float)
        
        # 最后根据输入和权重的小数点位置调整output大小
        output = output.mul(2 ** (input_s + self.weight_s))
        return output[:, 0:self.original_out_size]

    # int类型tensor转float，再进行Cuda中的einsum计算
    @staticmethod
    def tensor_int_einsum_cuda(rule:str, tensor0, tensor1) -> Tensor:
        if not tensor0.is_cuda or not tensor1.is_cuda:
            return torch.einsum(rule, tensor0, tensor1)
        
        float_tensor0 = tensor0.to(dtype=torch.float)
        float_tensor1 = tensor1.to(dtype=torch.float)
        einsum_result = torch.einsum(rule, float_tensor0, float_tensor1).to(dtype=torch.int64)

        return einsum_result
    