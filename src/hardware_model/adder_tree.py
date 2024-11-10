import configparser as cp
import math
from src.hardware_model.basic_component.adder import Adder

# 流水加法树，累加各xbar的OU计算结果
class AdderTree_PE:
    def __init__(self, SimConfig_path):
        addertree_config = cp.ConfigParser()
        addertree_config.read(SimConfig_path, encoding='UTF-8')
        self.xbar_group_num = int(addertree_config.get('Process element level', 'Group_Num'))
        self.ou_size = list(map(int, addertree_config.get('Crossbar level', 'OU_Size').split(',')))
        self.ou_row = int(self.ou_size[0])
        self.ou_column = int(self.ou_size[1])
        self.adder_model = Adder(SimConfig_path)

        self.adder_num = 0

        # from MNSIM，似乎有点蠢，不过懒得改了
        temp = self.xbar_group_num
        while temp/2 >= 1:
            self.adder_num += int(temp/2)
            temp = int(temp/2) + temp%2
        self.adder_num *= self.ou_column
        
        self.addertree_layer_num = math.ceil(math.log2(self.xbar_group_num))

        self.addertree_area = 0
        self.addertree_latency = 0
        self.addertree_energy = 0
        self.addertree_power = 0

        # self.caculate_addertree_power()

    def calculate_addertree_area(self):
        self.adder_model.calculate_adder_area()
        self.addertree_area = self.adder_model.adder_area * self.adder_num

    def calculate_addertree_latency(self):
        # self.adder_model.calculate_adder_latency()
        self.addertree_latency = self.adder_model.adder_latency * self.addertree_layer_num

    def calculate_addertree_energy(self):
        # 目前假设PE中的所有xbar都会被使用，不会空闲
        self.adder_model.calculate_adder_energy()
        self.addertree_energy = self.adder_model.adder_energy * self.adder_num

    def caculate_addertree_power(self):
        self.addertree_power = self.addertree_energy / self.addertree_latency
