import pickle
import torch
import src.nn_modules.matMulManager as mmm


class FpDataAnalyzer_Spec:
    def __init__(self, mm_manager=None) -> None:
        self.mm_manager = mm_manager
        assert mm_manager != None

        self.input_bit_dict_list = []
        self.ou_output_dict_list = []
        # self.ou_cycle_list = [[] for _ in range(mm_manager.weight_slice_num)]
        self.ou_cycle_list = []

        for i in range(mm_manager.input_slice_num):
            self.input_bit_dict_list.append([])
            for r in range(mm_manager.ou_row_num):
                self.input_bit_dict_list[i].append({})

        # 0:pp, 1:pn, 2:np, 3:nn
        for _ in range(2):
            self.ou_output_dict_list.append([])

        for i in range(mm_manager.input_slice_num):
            self.ou_output_dict_list[0].append([])
            self.ou_output_dict_list[1].append([])
            for j in range(mm_manager.weight_slice_num):
                self.ou_output_dict_list[0][i].append([])
                self.ou_output_dict_list[1][i].append([])
                for r in range(mm_manager.ou_row_num):
                    self.ou_output_dict_list[0][i][j].append([])
                    self.ou_output_dict_list[1][i][j].append([])
                    for c in range(mm_manager.ou_col_num):
                        self.ou_output_dict_list[0][i][j][r].append({})
                        self.ou_output_dict_list[1][i][j][r].append({})


        for s in range(2):
            self.ou_cycle_list.append([])
            for w in range(mm_manager.weight_slice_num):
                self.ou_cycle_list[s].append([])
                for r in range(mm_manager.ou_row_num):
                    self.ou_cycle_list[s][w].append([])
                    for c in range(mm_manager.ou_col_num):
                        self.ou_cycle_list[s][w][r].append([])
                        

    def update_ou_input(self, input_bits, in_b, ou_row):
        # 因为元素只有0和1，所以先沿着最内层维度求和，即计算每个子向量中的'1'数量，然后展平为一维
        ones_num_counts = input_bits.sum(dim=1)
        # 使用torch.unique统计不同'1'数量的出现频率
        ones_num, counts = torch.unique(ones_num_counts, return_counts=True)
        # 更新bit_vec_ones_num_counts字典
        for n, c in zip(ones_num, counts):
            if self.input_bit_dict_list.__contains__(n.item()):
                self.input_bit_dict_list[in_b][ou_row][n.item()] += c.item()
            else:
                self.input_bit_dict_list[in_b][ou_row][n.item()] = c.item()

    def update_ou_output(self, partial_out, sign, in_b, w_b, ou_row, ou_col):
        out_value, counts = torch.unique(partial_out, return_counts=True)
        for n, c in zip(out_value, counts):
            if self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col].__contains__(n.item()):
                self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col][n.item()] += c.item()
            else:
                self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col][n.item()] = c.item()
        
    def update_ou_output_cycle(self, partial_out_list, sign, w_b, ou_row, ou_col):
        b_size = len(partial_out_list[0])
        # print(b_size)
        for b in range(b_size):
            for i, partial_out in enumerate(partial_out_list):
                # print(f"i:{i}")
                # print(partial_out[b][0:15])
                self.ou_cycle_list[sign][w_b][ou_row][ou_col] += partial_out[b]

    def print_statistic(self):
        pass

    def save_statistic(self, save_path:str=None, postfix:str="test"):
        assert save_path != None and postfix != None
        if postfix == "test" or postfix == "sample":
            # layer_no, sign, in_slice, w_slice, in_pos, out_pos
            save_file = "/layer" + str(self.mm_manager.layer_no) + "_spec_xbr" + str(self.mm_manager.ou_size[0]) + \
                "_cellbit" + str(self.mm_manager.weight_slice_bit) + "_" + postfix + "_xbr_output.pkl"
            with open(save_path+save_file, "wb") as f:
                pickle.dump(self.ou_output_dict_list, f)
        elif postfix == "input":
            # layer_no, sign, in_slice, w_slice, in_pos, out_pos
            save_file = "/layer" + str(self.mm_manager.layer_no) + "_spec_xbr" + str(self.mm_manager.ou_size[0]) + "_" + postfix + ".pkl"
            with open(save_path+save_file, "wb") as f:
                pickle.dump(self.input_bit_dict_list, f)
        elif postfix == "cycle":
            save_file = "/layer" + str(self.mm_manager.layer_no) + "_spec_xbr" + str(self.mm_manager.ou_size[0]) + "_" + postfix + ".pkl"
            with open(save_path+save_file, "wb") as f:
                pickle.dump(self.ou_cycle_list, f)
        else:
            # layer_no, sign, in_slice, w_slice, in_pos, out_pos
            save_file = "/layer" + str(self.mm_manager.layer_no) + "_spec_xbr" + str(self.mm_manager.ou_size[0]) + \
                "_cellbit" + str(self.mm_manager.weight_slice_bit) + "_" + postfix + "_xbr_output.pkl"
            with open(save_path+save_file, "wb") as f:
                pickle.dump(self.ou_output_dict_list, f)


class FpDataAnalyzer_TRQ:
    def __init__(self, mm_manager:mmm.FixedPointMatMulManager_TRQ=None) -> None:
        self.mm_manager = mm_manager
        assert mm_manager != None

        self.ou_output_dict_list = []
        # 0:pp, 1:pn, 2:np, 3:nn
        for s in range(2):
            self.ou_output_dict_list.append([])
            for j in range(mm_manager.weight_slice_num):
                self.ou_output_dict_list[s].append({})

    def update_ou_output(self, partial_out, sign, w_b):
        out_value, counts = torch.unique(partial_out, return_counts=True)
        for n, c in zip(out_value, counts):
            if self.ou_output_dict_list[sign][w_b].__contains__(n.item()):
                self.ou_output_dict_list[sign][w_b][n.item()] += c.item()
            else:
                self.ou_output_dict_list[sign][w_b][n.item()] = c.item()
        

    def print_statistic(self):
        pass

    def save_statistic(self, save_path:str=None, postfix:str="test"):
        assert save_path != None and postfix != None
        
        # layer_no, sign, in_slice, w_slice, in_pos, out_pos
        save_file = "/layer" + str(self.mm_manager.layer_no) + "_trq_xbr" + str(self.mm_manager.ou_size[0]) + \
            "_cellbit" + str(self.mm_manager.weight_slice_bit) + "_" + postfix + "_xbr_output.pkl"
        # print(self.mm_manager.layer_no, save_path+save_file)
        
        with open(save_path+save_file, "wb") as f:
            pickle.dump(self.ou_output_dict_list, f)