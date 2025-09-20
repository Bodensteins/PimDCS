import math
import torch
from torch import Tensor
from opt_einsum import contract
from config.globalCfg import TensorType, globalCfg
import src.nn_modules.fixedPointArithmetic as fpA


class FixedPointMatMulManager:
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width:int,
                 input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                 is_slice_in_init:bool=True) -> None:
        # weight.shape)

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

        # 权重比特分片结果的三维Tensor，[bit_width, in_size, out_size]
        if is_slice_in_init:
            self.fp_weight_slices = FixedPointMatMulManager.slice_tensor(
                self.fp_weight, self.weight_bit_width, self.weight_slice_bit)

        # 输入暂存
        self.fp_input = None
        self.fp_input_slices = None

    # 矩阵比特分片，提供将每个切片拼接起来的选项
    # cat_dim=-1为不拼接，0为横向拼接，1为纵向拼接
    @staticmethod
    def slice_tensor(fp_tensor:Tensor, mat_bit_width:int, slice_bit:int=1, cat_dim:int=-1) -> Tensor:
        total_slice_num = math.ceil(mat_bit_width / slice_bit)
        fp_mat_slices = torch.zeros((total_slice_num, *fp_tensor.shape), dtype=torch.int8, device=fp_tensor.device)
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

    # 矩阵乘法
    def mat_mul(self, input:Tensor) -> Tensor:
        assert input.shape[1] == self.in_size
        assert input.device == self.device
        batch_size = input.shape[0]

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        self.fp_input = fp_input
        input_s = params[0].item()

        fp_input_slices = FixedPointMatMulManager.slice_tensor(fp_input, self.input_bit_width, self.input_slice_bit)
        self.fp_input_slices = fp_input_slices
        output = torch.zeros(batch_size, self.out_size, device=self.device)

        # 简单mat_mul暂时只考虑1bit分片
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

    # 只量化，不分片的矩阵乘法
    def fake_mat_mul(self, input:Tensor) -> Tensor:
        assert input.shape[1] == self.in_size
        assert input.device == self.device

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        self.fp_input = fp_input
        input_s = params[0].item()

        output = FixedPointMatMulManager.tensor_int_mul_cuda(fp_input, self.fp_weight).to(dtype=torch.float)
        output = output.mul(2 ** (input_s + self.weight_s))
        return output

    # 用cuda对两个int类型二维tensor进行乘法
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        if not mat0.is_cuda or not mat1.is_cuda:
            return torch.matmul(mat0, mat1)

        float_tensor0 = mat0.to(dtype=torch.float)
        float_tensor1 = mat1.to(dtype=torch.float)
        mul_result = float_tensor0.matmul(float_tensor1)

        return mul_result.to(dtype=torch.int64)


class FixedPointMatMulManager_Spec(FixedPointMatMulManager):
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width:int,
                xbr_size:tuple=globalCfg.crxShape, ou_size:tuple=globalCfg.ouShape, 
                input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                is_slice_in_init:bool=True, is_split_in_init:bool=True, 
                layer_no:int=0):
        
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

        self.layer_no = layer_no
        self.ou_row_num = math.ceil(self.in_size / ou_size[0])
        self.ou_col_num = math.ceil(self.out_size / ou_size[1])

        self.fp_weight_pos = torch.clamp(self.fp_weight, min=0)
        self.fp_weight_neg = -torch.clamp(self.fp_weight, max=0)

        # 拆分正负阵列，丢弃符号位
        self.weight_bit_width -= 1

        self.input_slice_num = math.ceil(self.input_bit_width / input_slice_bit)
        self.weight_slice_num = math.ceil(self.weight_bit_width / weight_slice_bit)

        # print(self.layer_no)
        # print(weight.shape, self.in_size, self.out_size, self.ou_size)
        # print(self.input_slice_num, self.weight_slice_num, self.ou_row_num, self.ou_col_num)

        self.fp_weight_slices_pos = None
        self.fp_weight_slices_neg = None

        # 输入和权重分片是否合并及合并维度
        # -1为不合并，0为横向合并，1为纵向合并
        self.input_slice_cat_dim = -1
        self.weight_slice_cat_dim = -1

        # 权重比特分片结果的二维Tensor，[bit_width, in_size, out_size]
        if is_slice_in_init:
            self.fp_weight_slices_pos = FixedPointMatMulManager_Spec.slice_tensor(
                self.fp_weight_pos, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)
            self.fp_weight_slices_neg = FixedPointMatMulManager_Spec.slice_tensor(
                self.fp_weight_neg, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)

        # self.fp_weight_split = None
        self.fp_weight_split_pos = None
        self.fp_weight_split_neg = None
        if is_split_in_init:
            self.fp_weight_split_pos = FixedPointMatMulManager_Spec.split_weight(self.fp_weight_slices_pos, ou_size)
            self.fp_weight_split_neg = FixedPointMatMulManager_Spec.split_weight(self.fp_weight_slices_neg, ou_size)

        self.input_s = None
        self.is_ignore_neg_input = False
        self.data_analyzer = None


    @staticmethod
    def split_weight(weight_slices:Tensor, split_size:tuple):
        slice_num, row_size, col_size = weight_slices.shape
        ou_row_size, ou_col_size = split_size
        assert row_size % ou_row_size == 0 and col_size % ou_col_size == 0
        
        ou_row_num = row_size // ou_row_size
        ou_col_num = col_size // ou_col_size

        weight_split = weight_slices.reshape(slice_num, ou_row_num, ou_row_size, ou_col_num, ou_col_size)
        weight_split = weight_split.permute(0, 1, 3, 2, 4)
        return weight_split

    @staticmethod
    def split_input(input_slices:Tensor, split_in_size:int):
        slice_num, batch_size, in_size = input_slices.shape
        assert in_size % split_in_size == 0

        split_in_num = in_size // split_in_size
        input_split = input_slices.reshape(slice_num, batch_size, split_in_num, split_in_size)
        input_split = input_split.permute(0, 2, 1, 3)

        return input_split
    
    # 不拆输入
    def mat_mul(self, input:Tensor) -> Tensor:
        # print(self.original_in_size, self.original_out_size)
        assert input.device == self.device
        assert input.shape[1] == self.original_in_size
        batch_size = input.shape[0]

        if input.min() < 0:
            self.is_ignore_neg_input = False
        else:
            self.is_ignore_neg_input = True

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        self.input_s = params[0].item()

        # 根据ou_row大小补全输入，使输入长度与ou_row大小对齐
        if self.original_in_size % self.ou_size[0] != 0:
            temp_size = self.ou_size[0] - self.original_in_size % self.ou_size[0]
            fp_input = torch.cat((fp_input, torch.zeros((batch_size, temp_size), dtype=fp_input.dtype, device=fp_input.device)), dim=1)
        self.fp_input = fp_input


        fp_input_slices = FixedPointMatMulManager_Spec.slice_tensor(fp_input, self.input_bit_width, self.input_slice_bit)
        # self.fp_input_slices = fp_input_slices

        fp_input_split = FixedPointMatMulManager_Spec.split_input(fp_input_slices, self.ou_size[0])

        output = torch.zeros(batch_size, self.out_size, device=self.device)

        # layer_no, sign, in_slice, w_slice, in_pos, out_pos
        # print(self.weight_slice_num)
        # 输入只支持1bit
        for i in range(self.input_slice_num):
            for w in range(self.weight_slice_num):
                for r in range(self.ou_row_num):
                    for c in range(self.ou_col_num):
                        weight_tensor_pos = self.fp_weight_split_pos[w, r, c, :, :]
                        weight_tensor_neg = self.fp_weight_split_neg[w, r, c, :, :]
                        input_tensor = fp_input_split[i, r, :, :]
                        partial_out_p = FixedPointMatMulManager_Spec.tensor_int_mul_cuda(input_tensor, weight_tensor_pos)
                        partial_out_n = FixedPointMatMulManager_Spec.tensor_int_mul_cuda(input_tensor, weight_tensor_neg)
                        partial_out = partial_out_p - partial_out_n
                        
                        # send to data_analyzer
                        self.data_analyzer.update_ou_output(partial_out_p, 0, i, w, r, c)
                        self.data_analyzer.update_ou_output(partial_out_n, 1, i, w, r, c)

                        if i == self.input_slice_num - 1:
                            partial_out = -partial_out
                        
                        in_b = i * self.input_slice_bit
                        w_b = w * self.weight_slice_bit
                        output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] = \
                            output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] + partial_out.mul(2 ** (in_b + w_b + self.input_s + self.weight_s))

        return output[:, 0:self.original_out_size]

    def fake_mat_mul(self, input: Tensor) -> Tensor:
        return self.mat_mul(input)
    
    @staticmethod
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        return FixedPointMatMulManager.tensor_int_mul_cuda(mat0, mat1)

    def set_data_analyzer(self, data_analyzer) -> None:
        self.data_analyzer = data_analyzer


class FixedPointMatMulManager_TRQ(FixedPointMatMulManager):
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width:int,
                xbr_size:tuple=globalCfg.crxShape, ou_size:tuple=globalCfg.ouShape, 
                input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                is_slice_in_init:bool=True, is_split_in_init:bool=True, 
                layer_no:int=0):

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

        self.layer_no = layer_no
        self.ou_row_num = math.ceil(self.in_size / ou_size[0])

        self.fp_weight_pos = torch.clamp(self.fp_weight, min=0)
        self.fp_weight_neg = -torch.clamp(self.fp_weight, max=0)

        # 拆分正负阵列，丢弃符号位
        self.weight_bit_width -= 1

        self.input_slice_num = math.ceil(self.input_bit_width / input_slice_bit)
        self.weight_slice_num = math.ceil(self.weight_bit_width / weight_slice_bit)

        # print(self.layer_no)
        # print(weight.shape, self.in_size, self.out_size, self.ou_size)
        # print(self.input_slice_num, self.weight_slice_num, self.ou_row_num, self.ou_col_num)

        self.fp_weight_slices_pos = None
        self.fp_weight_slices_neg = None

        # 输入和权重分片是否合并及合并维度
        # -1为不合并，0为横向合并，1为纵向合并
        self.input_slice_cat_dim = -1
        self.weight_slice_cat_dim = -1

        # 权重比特分片结果的二维Tensor，[bit_width, in_size, out_size]
        if is_slice_in_init:
            self.fp_weight_slices_pos = FixedPointMatMulManager_TRQ.slice_tensor(
                self.fp_weight_pos, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)
            self.fp_weight_slices_neg = FixedPointMatMulManager_TRQ.slice_tensor(
                self.fp_weight_neg, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)

        # self.fp_weight_split = None
        self.fp_weight_split_pos = None
        self.fp_weight_split_neg = None
        if is_split_in_init:
            self.fp_weight_split_pos = FixedPointMatMulManager_TRQ.split_weight(self.fp_weight_slices_pos, ou_size)
            self.fp_weight_split_neg = FixedPointMatMulManager_TRQ.split_weight(self.fp_weight_slices_neg, ou_size)

        self.input_s = None
        self.is_ignore_neg_input = False
        self.data_analyzer = None


    @staticmethod
    def split_weight(weight_slices:Tensor, split_size:tuple):
        slice_num, row_size, col_size = weight_slices.shape
        ou_row_size, _ = split_size
        assert row_size % ou_row_size == 0
        
        ou_row_num = row_size // ou_row_size

        weight_split = weight_slices.reshape(slice_num, ou_row_num, ou_row_size, col_size)
        # weight_split = weight_split.permute(0, 1, 3, 2, 4)
        return weight_split

    @staticmethod
    def split_input(input_slices:Tensor, split_in_size:int):
        slice_num, batch_size, in_size = input_slices.shape
        assert in_size % split_in_size == 0

        split_in_num = in_size // split_in_size
        input_split = input_slices.reshape(slice_num, batch_size, split_in_num, split_in_size)
        input_split = input_split.permute(0, 2, 1, 3)

        return input_split

    def mat_mul(self, input: Tensor) -> Tensor:
        # print(self.layer_no)

        assert input.device == self.device
        assert input.shape[1] == self.original_in_size
        batch_size = input.shape[0]

        if input.min() < 0:
            self.is_ignore_neg_input = False
        else:
            self.is_ignore_neg_input = True

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width, TensorType.Normal)
        self.input_s = params[0].item()

        # 根据ou_row大小补全输入，使输入长度与ou_row大小对齐
        if self.original_in_size % self.ou_size[0] != 0:
            temp_size = self.ou_size[0] - self.original_in_size % self.ou_size[0]
            fp_input = torch.cat((fp_input, torch.zeros((batch_size, temp_size), dtype=fp_input.dtype, device=fp_input.device)), dim=1)
        self.fp_input = fp_input

        fp_input_slices = FixedPointMatMulManager_TRQ.slice_tensor(fp_input, self.input_bit_width, self.input_slice_bit)
        # self.fp_input_slices = fp_input_slices

        fp_input_split = FixedPointMatMulManager_TRQ.split_input(fp_input_slices, self.ou_size[0])

        output = torch.zeros(batch_size, self.out_size, device=self.device)

        # layer_no, sign, in_slice, w_slice, in_pos, out_pos
        # print(self.weight_slice_num)
        # 输入只支持1bit
        for i in range(self.input_slice_num):
            for w in range(self.weight_slice_num):
                for r in range(self.ou_row_num):
                    weight_tensor_pos = self.fp_weight_split_pos[w, r, :, :]
                    weight_tensor_neg = self.fp_weight_split_neg[w, r, :, :]
                    input_tensor = fp_input_split[i, r, :, :]
                    partial_out_p = FixedPointMatMulManager_TRQ.tensor_int_mul_cuda(input_tensor, weight_tensor_pos)
                    partial_out_n = FixedPointMatMulManager_TRQ.tensor_int_mul_cuda(input_tensor, weight_tensor_neg)
                    partial_out = partial_out_p - partial_out_n
                    
                    # send to data_analyzer
                    self.data_analyzer.update_ou_output(partial_out_p, 0, w)
                    self.data_analyzer.update_ou_output(partial_out_n, 1, w)

                    if i == self.input_slice_num - 1:
                        partial_out = -partial_out
                    
                    in_b = i * self.input_slice_bit
                    w_b = w * self.weight_slice_bit
                    output += partial_out.mul(2 ** (in_b + w_b + self.input_s + self.weight_s))

        return output[:, 0:self.original_out_size]
    
    def fake_mat_mul(self, input: Tensor) -> Tensor:
        return self.mat_mul(input)
    
    @staticmethod
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        return FixedPointMatMulManager.tensor_int_mul_cuda(mat0, mat1)

    def set_data_analyzer(self, data_analyzer) -> None:
        self.data_analyzer = data_analyzer


class FixedPointMatMulManager_Tailor(FixedPointMatMulManager):
    def __init__(self, weight:Tensor, weight_bit_width:int, input_bit_width:int,
                xbr_size:tuple=globalCfg.crxShape, ou_size:tuple=globalCfg.ouShape, 
                input_slice_bit:int=1, weight_slice_bit:int=globalCfg.cellBits,
                is_slice_in_init:bool=True, is_split_in_init:bool=True, 
                layer_no:int=0, adc_resolution:int=8, n_noise_max:int=11):

        self.adc_resolution = adc_resolution
        self.n_noise_max = n_noise_max

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

        self.layer_no = layer_no
        self.ou_row_num = math.ceil(self.in_size / ou_size[0])
        self.ou_col_num = math.ceil(self.out_size / ou_size[1])

        self.fp_weight_pos = torch.clamp(self.fp_weight, min=0)
        self.fp_weight_neg = -torch.clamp(self.fp_weight, max=0)

        # 拆分正负阵列，丢弃符号位
        self.weight_bit_width -= 1

        self.input_slice_num = math.ceil(self.input_bit_width / input_slice_bit)
        self.weight_slice_num = math.ceil(self.weight_bit_width / weight_slice_bit)

        # print(self.layer_no)
        # print(weight.shape, self.in_size, self.out_size, self.ou_size)
        # print(self.input_slice_num, self.weight_slice_num, self.ou_row_num, self.ou_col_num)

        self.fp_weight_slices_pos = None
        self.fp_weight_slices_neg = None

        # 输入和权重分片是否合并及合并维度
        # -1为不合并，0为横向合并，1为纵向合并
        self.input_slice_cat_dim = -1
        self.weight_slice_cat_dim = -1

        # 权重比特分片结果的二维Tensor，[bit_width, in_size, out_size]
        if is_slice_in_init:
            self.fp_weight_slices_pos = FixedPointMatMulManager_Tailor.slice_tensor(
                self.fp_weight_pos, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)
            self.fp_weight_slices_neg = FixedPointMatMulManager_Tailor.slice_tensor(
                self.fp_weight_neg, self.weight_bit_width, self.weight_slice_bit, cat_dim=self.weight_slice_cat_dim)

        # self.fp_weight_split = None
        self.fp_weight_split_pos = None
        self.fp_weight_split_neg = None
        if is_split_in_init:
            self.fp_weight_split_pos = FixedPointMatMulManager_Tailor.split_weight(self.fp_weight_slices_pos, ou_size)
            self.fp_weight_split_neg = FixedPointMatMulManager_Tailor.split_weight(self.fp_weight_slices_neg, ou_size)

        self.input_s = None
        self.is_ignore_neg_input = False
        self.data_analyzer = None


    @staticmethod
    def split_weight(weight_slices:Tensor, split_size:tuple):
        slice_num, row_size, col_size = weight_slices.shape
        ou_row_size, ou_col_size = split_size
        assert row_size % ou_row_size == 0 and col_size % ou_col_size == 0
        
        ou_row_num = row_size // ou_row_size
        ou_col_num = col_size // ou_col_size

        weight_split = weight_slices.reshape(slice_num, ou_row_num, ou_row_size, ou_col_num, ou_col_size)
        weight_split = weight_split.permute(0, 1, 3, 2, 4)
        return weight_split

    @staticmethod
    def split_input(input_slices:Tensor, split_in_size:int):
        slice_num, batch_size, in_size = input_slices.shape
        assert in_size % split_in_size == 0

        split_in_num = in_size // split_in_size
        input_split = input_slices.reshape(slice_num, batch_size, split_in_num, split_in_size)
        input_split = input_split.permute(0, 2, 1, 3)

        return input_split
    
    def tailor(self, out_slice:Tensor, i, w):
        in_b = i * self.input_slice_bit
        w_b = w * self.weight_slice_bit
        noise_max = 1 << self.n_noise_max
        noise_alloc = noise_max // (self.input_bit_width * self.weight_bit_width)
        if noise_alloc < 2:
            redundant_bit = 0
        else:
            redundant_bit = math.floor(math.log2(noise_alloc)) - in_b - w_b
        # print(redundant_bit)
        if redundant_bit <= 0:
            return out_slice
        if redundant_bit >= self.adc_resolution:
            out_slice.zero_()
            return out_slice
        max_point = 1 << self.adc_resolution
        max_mask = out_slice > max_point
        out_slice[max_mask] = max_point
        out_slice = out_slice & ~((1 << redundant_bit) - 1)
        return out_slice

    def mat_mul(self, input: Tensor) -> Tensor:
        # print(self.layer_no)

        assert input.device == self.device
        assert input.shape[1] == self.original_in_size
        batch_size = input.shape[0]

        if input.min() < 0:
            self.is_ignore_neg_input = False
        else:
            self.is_ignore_neg_input = True

        # 输入量化
        fp_input, params = fpA.quantization_tensor(input, self.input_bit_width + 1, TensorType.Normal)
        self.input_s = params[0].item()

        # 根据ou_row大小补全输入，使输入长度与ou_row大小对齐
        if self.original_in_size % self.ou_size[0] != 0:
            temp_size = self.ou_size[0] - self.original_in_size % self.ou_size[0]
            fp_input = torch.cat((fp_input, torch.zeros((batch_size, temp_size), dtype=fp_input.dtype, device=fp_input.device)), dim=1)
        self.fp_input = fp_input

        # 拆分为正负输入
        fp_input_pos = torch.clamp(self.fp_input, min=0)
        fp_input_neg = -torch.clamp(self.fp_input, max=0)

        fp_input_slices_pos = FixedPointMatMulManager_Tailor.slice_tensor(fp_input_pos, self.input_bit_width, self.input_slice_bit)
        fp_input_slices_neg = FixedPointMatMulManager_Tailor.slice_tensor(fp_input_neg, self.input_bit_width, self.input_slice_bit)
        # self.fp_input_slices = fp_input_slices

        fp_input_split_pos = FixedPointMatMulManager_Tailor.split_input(fp_input_slices_pos, self.ou_size[0])
        fp_input_split_neg = FixedPointMatMulManager_Tailor.split_input(fp_input_slices_neg, self.ou_size[0])

        output = torch.zeros(batch_size, self.out_size, device=self.device)

        # layer_no, sign, in_slice, w_slice, in_pos, out_pos
        for i in range(self.input_slice_num):
            for w in range(self.weight_slice_num):
                for r in range(self.ou_row_num):
                    for c in range(self.ou_col_num):
                        weight_tensor_pos = self.fp_weight_split_pos[w, r, c, :, :]
                        weight_tensor_neg = self.fp_weight_split_neg[w, r, c, :, :]
                        input_tensor_pos = fp_input_split_pos[i, r, :, :]
                        partial_out_pp = FixedPointMatMulManager_Tailor.tensor_int_mul_cuda(input_tensor_pos, weight_tensor_pos)
                        partial_out_pn = FixedPointMatMulManager_Tailor.tensor_int_mul_cuda(input_tensor_pos, weight_tensor_neg)
                        # print(partial_out_pp)
                        partial_out_pp = self.tailor(partial_out_pp, i, w)
                        partial_out_pn = self.tailor(partial_out_pn, i, w)
                        partial_out = partial_out_pp - partial_out_pn

                        if not self.is_ignore_neg_input:
                            input_tensor_neg = fp_input_split_neg[i, r, :, :]
                            partial_out_np = FixedPointMatMulManager_Tailor.tensor_int_mul_cuda(input_tensor_neg, weight_tensor_pos)
                            partial_out_nn = FixedPointMatMulManager_Tailor.tensor_int_mul_cuda(input_tensor_neg, weight_tensor_neg)
                            partial_out_np = self.tailor(partial_out_np, i, w)
                            partial_out_nn = self.tailor(partial_out_nn, i, w)
                            partial_out = partial_out - partial_out_np + partial_out_nn
                        
                        in_b = i * self.input_slice_bit
                        w_b = w * self.weight_slice_bit
                        output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] = \
                            output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] + partial_out.mul(2 ** (in_b + w_b + self.input_s + self.weight_s))

        return output[:, 0:self.original_out_size]
    
    def fake_mat_mul(self, input: Tensor) -> Tensor:
        return self.mat_mul(input)
    
    @staticmethod
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        return FixedPointMatMulManager.tensor_int_mul_cuda(mat0, mat1)

    def set_data_analyzer(self, data_analyzer) -> None:
        self.data_analyzer = data_analyzer


