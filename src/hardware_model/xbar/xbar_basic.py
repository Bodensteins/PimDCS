import configparser as cp
from src.hardware_model.basic_component.device import Device


class Crossbar_Basic:
    # 目前只支持1T1R单元和行为级读（计算）模拟

    def __init__(self, SimConfig_path):
        # device.__init__(self,SimConfig_path)
        xbar_config = cp.ConfigParser()
        xbar_config.read(SimConfig_path, encoding='UTF-8')
        self.xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = int(self.xbar_size[0])
        self.xbar_column = int(self.xbar_size[1])
        # self.subarray_size = int(xbar_config.get('Crossbar level', 'Subarray_Size'))
        # assert self.xbar_row % self.subarray_size == 0, "The crossbar size must be divisible by the subarray size"
        # self.subarray_num = self.xbar_row/self.subarray_size
        self.PIM_type = int(xbar_config.get('Process element level', 'PIM_Type'))
        self.cell_type = xbar_config.get('Crossbar level', 'Cell_Type')
        self.transistor_tech = int(xbar_config.get('Crossbar level', 'Transistor_Tech'))
        self.wire_resistance = float(xbar_config.get('Crossbar level', 'Wire_Resistance'))
        self.wire_capacity = float(xbar_config.get('Crossbar level', 'Wire_Capacity'))
        self.area_calculation_method = int(xbar_config.get('Crossbar level', 'Area_Calculation'))
        self.xbar_area = 0
        # self.xbar_simulation_level = int(xbar_config.get('Algorithm Configuration', 'Simulation_Level'))

        self.device_model = Device(SimConfig_path)

        self.xbar_read_power = 0
        self.xbar_read_latency = 0

        # self.xbar_write_power = 0
        # self.xbar_write_latency = 0

        self.xbar_read_energy = 0
        # self.xbar_write_energy = 0

        # self.xbar_num_write_row = 0
        # self.xbar_num_write_column = 0

        # self.xbar_num_read_row = 0
        # self.xbar_num_read_column = 0

        self.ou_size = list(map(int, xbar_config.get('Crossbar level', 'OU_Size').split(',')))
        self.ou_row = int(self.ou_size[0])
        self.ou_column = int(self.ou_size[1])
        assert self.xbar_row % self.ou_row == 0 and self.xbar_column % self.ou_column == 0, "The crossbar size must be divisible by the OU size"

        self.ou_num = (self.xbar_row // self.ou_row) * (self.xbar_column // self.ou_column)

        self.calculate_xbar_read_power()

    def calculate_xbar_area(self):
        # 使用器件工艺制程计算面积
        WL_ratio = 3
        # WL_ratio is the technology parameter W/L of the transistor
        self.xbar_area = 3 * (WL_ratio + 1) * self.xbar_row * self.xbar_column * self.device_tech**2 * 1e-6
        
    def calculate_xbar_read_latency(self):
        # 计算一次OU读（计算）的延迟
        size = self.xbar_row*self.xbar_column / 1024 / 8  # KB
        wire_latency = 0.001 * (0.0002 * size ** 2 + 5 * 10 ** -6 * size + 4 * 10 ** -14)  # ns，wire latency不是很精确，不过影响很小
        self.xbar_read_latency = self.device_model.device_read_latency + wire_latency
        
    def calculate_xbar_read_power(self):
        # 计算一次OU的功率
        self.device_model.calculate_device_read_power()
        self.xbar_read_power = self.ou_row * self.ou_column * self.device_model.device_read_power

    def calculate_xbar_read_energy(self):
        # unit: nJ
        # 一次OU计算的能耗
        self.xbar_read_energy = self.xbar_read_power * self.xbar_read_latency

    