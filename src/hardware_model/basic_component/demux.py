# from MNSIM

import configparser as cp


# todo: Demux需要重新建模
class Demux:
    def __init__(self, SimConfig_path, demux_out_num=8):
        demux_config = cp.ConfigParser()
        demux_config.read(SimConfig_path, encoding='UTF-8')
        self.transistor_tech = int(demux_config.get('Crossbar level', 'Transistor_Tech'))
        self.demux_out_num = demux_out_num

        self.demux_area = 0
        self.demux_latency = 0
        self.demux_power = 0
        self.demux_energy = 0

        self.calculate_demux_power()

    def calculate_demux_area(self):
        transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
        demux_area_dict = {2: 8*transistor_area, # 2-1: 8 transistors
                           4: 24*transistor_area, # 4-1: 3 * 2-1
                           8: 72*transistor_area,
                           16: 216*transistor_area,
                           32: 648*transistor_area,
                           64: 1944*transistor_area
        }
        # unit: um^2
        if self.demux_out_num <= 2:
            self.demux_area = demux_area_dict[2]
        elif self.demux_out_num<=4:
            self.demux_area = demux_area_dict[4]
        elif self.demux_out_num<=8:
            self.demux_area = demux_area_dict[8]
        elif self.demux_out_num<=16:
            self.demux_area = demux_area_dict[16]
        elif self.demux_out_num<=32:
            self.demux_area = demux_area_dict[32]
        else:
            self.demux_area = demux_area_dict[64]

    # todo: 补全latency
    def calculate_demux_latency(self):
        # todo
        demux_latency_dict = {
            1:0.27933 # 1:8, technology 65nm
        }

        self.demux_latency = demux_latency_dict[1]


    def calculate_demux_power(self):
        transistor_power = 10*1.2/1e9
        demux_power_dict = {2: 8*transistor_power,
                         4: 24*transistor_power,
                         8: 72*transistor_power,
                         16: 216*transistor_power,
                         32: 648*transistor_power,
                         64: 1944*transistor_power
        }
        # unit: W
        if self.demux_out_num <= 2:
            self.input_demux_power = demux_power_dict[2]
        elif self.demux_out_num<=4:
            self.input_demux_power = demux_power_dict[4]
        elif self.demux_out_num<=8:
            self.input_demux_power = demux_power_dict[8]
        elif self.demux_out_num<=16:
            self.input_demux_power = demux_power_dict[16]
        elif self.demux_out_num<=32:
            self.input_demux_power = demux_power_dict[32]
        else:
            self.input_demux_power = demux_power_dict[64]

    # todo
    def calculate_demux_energy(self):
        self.demux_energy = self.demux_latency * self.demux_power

