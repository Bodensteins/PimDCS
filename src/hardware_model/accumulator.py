import configparser as cp
# import math
from src.hardware_model.basic_component.adder import Adder

class Accumulator_PE:
    def __init__(self, SimConfig_path):
        accumulator_config = cp.ConfigParser()
        accumulator_config.read(SimConfig_path, encoding='UTF-8')
        self.xbar_group_num = int(accumulator_config.get('Process element level', 'Group_Num'))
        self.ou_size = list(map(int, accumulator_config.get('Crossbar level', 'OU_Size').split(',')))
        self.ou_row = int(self.ou_size[0])
        self.ou_column = int(self.ou_size[1])

        self.adder_model = Adder(SimConfig_path)

        self.adder_num = self.ou_column

        self.accumulator_area = 0
        self.accumulator_latency = 0
        self.accumulator_energy = 0
        self.accumulator_power = 0

    def calculate_accumulator_area(self):
        self.adder_model.calculate_adder_area()
        self.accumulator_area = self.adder_model.adder_area * self.adder_num

    def calculate_accumulator_latency(self):
        # self.adder_model.calculate_adder_latency()
        self.accumulator_latency = self.adder_model.adder_latency

    def calculate_accumulator_energy(self):
        self.adder_model.calculate_adder_energy()
        self.accumulator_energy = self.adder_model.adder_energy * self.adder_num

    def calculate_accumulator_power(self):
        self.accumulator_power = self.accumulator_energy / self.accumulator_latency

