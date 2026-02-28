import math
import torch
from torch import Tensor
from config.globalCfg import TensorType, globalCfg
from src.nn_modules.matMulManager import FixedPointMatMulManager
import src.nn_modules.fixedPointArithmetic as fpA

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
        # self.is_ignore_neg_input = False
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
        for w in range(self.weight_slice_num):
            for r in range(self.ou_row_num):
                for c in range(self.ou_col_num):
                    partial_out_list_p = []
                    partial_out_list_n = []
                    for i in range(self.input_slice_num):
                        weight_tensor_pos = self.fp_weight_split_pos[w, r, c, :, :]
                        weight_tensor_neg = self.fp_weight_split_neg[w, r, c, :, :]
                        input_tensor = fp_input_split[i, r, :, :]
                        partial_out_p = FixedPointMatMulManager_Spec.tensor_int_mul_cuda(input_tensor, weight_tensor_pos)
                        partial_out_n = FixedPointMatMulManager_Spec.tensor_int_mul_cuda(input_tensor, weight_tensor_neg)
                        partial_out = partial_out_p - partial_out_n
                        
                        # send to data_analyzer
                        # self.data_analyzer.update_ou_output(partial_out_p, 0, i, w, r, c)
                        # self.data_analyzer.update_ou_output(partial_out_n, 1, i, w, r, c)

                        # print(partial_out_p.shape)
                        partial_out_list_p.append(partial_out_p.tolist())
                        partial_out_list_n.append(partial_out_n.tolist())
                        
                        if i == self.input_slice_num - 1:
                            partial_out = -partial_out
                        
                        in_b = i * self.input_slice_bit
                        w_b = w * self.weight_slice_bit
                        output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] = \
                            output[:, c*self.ou_size[1]:(c+1)*self.ou_size[1]] + partial_out.mul(2 ** (in_b + w_b + self.input_s + self.weight_s))

                    self.data_analyzer.update_ou_output_cycle(partial_out_list_p, 0, w, r, c)
                    self.data_analyzer.update_ou_output_cycle(partial_out_list_n, 1, w, r, c)
        # for i in range(self.input_slice_num):
        #         for r in range(self.ou_row_num):
        #             input_tensor = fp_input_split[i, r, :, :]
        #             self.data_analyzer.update_ou_input(input_tensor, i, r)

        return output[:, 0:self.original_out_size]

    def fake_mat_mul(self, input: Tensor) -> Tensor:
        return self.mat_mul(input)
    
    @staticmethod
    def tensor_int_mul_cuda(mat0, mat1) -> Tensor:
        return FixedPointMatMulManager.tensor_int_mul_cuda(mat0, mat1)

    def set_data_analyzer(self, data_analyzer) -> None:
        self.data_analyzer = data_analyzer