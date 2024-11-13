# from MNSIM

import configparser as cp


# todo: mux需要重新建模
class Mux:
    def __init__(self, SimConfig_path, mux_in_num=8):
        demux_config = cp.ConfigParser()
        demux_config.read(SimConfig_path, encoding='UTF-8')
        self.transistor_tech = int(demux_config.get('Crossbar level', 'Transistor_Tech'))
        self.mux_in_num = mux_in_num

        self.mux_area = 0
        self.mux_latency = 0
        self.mux_power = 0
        self.mux_energy = 0

        self.calculate_mux_power()

    def calculate_mux_area(self):
        transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
        mux_area_dict = {2: 8*transistor_area,
                         4: 24*transistor_area,
                         8: 72*transistor_area,
                         16: 216*transistor_area,
                         32: 648*transistor_area,
                         64: 1944*transistor_area
        }
        # unit: um^2
        if self.mux_in_num <= 2:
            self.mux_area = mux_area_dict[2]
        elif self.mux_in_num <= 4:
            self.mux_area = mux_area_dict[4]
        elif self.mux_in_num <= 8:
            self.mux_area = mux_area_dict[8]
        elif self.mux_in_num <= 16:
            self.mux_area = mux_area_dict[16]
        elif self.mux_in_num <= 32:
            self.mux_area = mux_area_dict[32]
        else:
            self.mux_area = mux_area_dict[64]

    def calculate_mux_power(self):
        transistor_power = 10*1.2/1e9
        mux_power_dict = {2: 8*transistor_power,
                         4: 24*transistor_power,
                         8: 72*transistor_power,
                         16: 216*transistor_power,
                         32: 648*transistor_power,
                         64: 1944*transistor_power
        }
        # unit: W
        if self.mux_in_num <= 2:
            self.mux_power = mux_power_dict[2]
        elif self.mux_in_num <= 4:
            self.mux_power = mux_power_dict[4]
        elif self.mux_in_num <= 8:
            self.mux_power = mux_power_dict[8]
        elif self.mux_in_num <= 16:
            self.mux_power = mux_power_dict[16]
        elif self.mux_in_num <= 32:
            self.mux_power = mux_power_dict[32]
        else:
            self.mux_power = mux_power_dict[64]

    def calculate_mux_latency(self):
        muxLatency_dict = {
            1:32.744/1000   # ns
        }

        self.mux_latency = muxLatency_dict[1]

    def calculate_mux_energy(self):
        assert self.mux_power >= 0
        assert self.mux_latency >= 0
        self.mux_energy = self.mux_power * self.mux_power
