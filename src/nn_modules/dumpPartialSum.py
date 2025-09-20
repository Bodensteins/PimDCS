import pickle
import torch
import src.nn_modules.matMulManager as mmm


class FpDataAnalyzer_Spec:
    def __init__(self, mm_manager:mmm.FixedPointMatMulManager_Spec=None) -> None:
        self.mm_manager = mm_manager
        assert mm_manager != None

        self.ou_output_dict_list = []

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

    def update_ou_output(self, partial_out, sign, in_b, w_b, ou_row, ou_col):
        out_value, counts = torch.unique(partial_out, return_counts=True)
        for n, c in zip(out_value, counts):
            if self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col].__contains__(n.item()):
                self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col][n.item()] += c.item()
            else:
                self.ou_output_dict_list[sign][in_b][w_b][ou_row][ou_col][n.item()] = c.item()
        

    def print_statistic(self):
        pass

    def save_statistic(self, save_path:str=None, postfix:str="test"):
        assert save_path != None and postfix != None
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