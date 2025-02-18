import configparser as cp
import os
import math

# 需修改为多bit cell情况
class Device:
    def __init__(self, SimConfig_path):
        device_config = cp.ConfigParser()
        device_config.read(SimConfig_path, encoding='UTF-8')
        self.device_tech = float(device_config.get('Device level', 'Device_Tech'))
        # self.device_type = device_config.get('Device level', 'Device_Type')
        
        self.device_area = float(device_config.get('Device level', 'Device_Area'))
        self.device_read_voltage_level = int(device_config.get('Device level', 'Read_Level'))
        assert self.device_read_voltage_level >= 0, "Read voltage level < 0"
        self.device_read_voltage = list(map(float, device_config.get('Device level', 'Read_Voltage').split(',')))
        assert self.device_read_voltage_level == len(self.device_read_voltage), "Read voltage setting error"

        self.device_write_voltage_level = int(device_config.get('Device level', 'Write_Level'))
        assert self.device_write_voltage_level >= 0, "Write voltage level < 0"
        self.device_write_voltage = list(map(float, device_config.get('Device level', 'Write_Voltage').split(',')))
        assert self.device_write_voltage_level == len(self.device_write_voltage), "Write voltage setting error"

        self.device_read_latency = float(device_config.get('Device level', 'Read_Latency'))
        self.device_write_latency = float(device_config.get('Device level', 'Write_Latency'))

        self.device_level = int(device_config.get('Device level', 'Device_Level'))
        assert self.device_level >= 0, "NVM resistance level < 0"
        self.device_resistance = list(map(float, device_config.get('Device level', 'Device_Resistance').split(',')))
        assert self.device_level == len(self.device_resistance), "NVM resistance setting error"

        # self.decice_variation = float(device_config.get('Device level', 'Device_Variation'))
        # Device variation is defined as \Delta R / R

        self.device_read_power = 0
        self.device_write_power = 0
        # print("Device configuration is loaded")
        self.device_read_energy = 0
        self.device_write_energy = 0

    # 一个单元的读功率
    def calculate_device_read_power(self, R = None, V = None):
        # R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
        if R is None:
            # R = math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
            # R = float(self.device_resistance[-1]) #worst case estimation
            # R = 0.75*float(self.device_resistance[0]) + 0.25*float(self.device_resistance[-1])
            R = (float(self.device_resistance[0])*float(self.device_resistance[-1]))/\
                (float(self.device_resistance[-1])*0.67+float(self.device_resistance[0])*0.33)
        assert R > 0, "Resistance <= 0"
        if V is None:
            # V = math.sqrt((self.device_read_voltage[0]**2 + self.device_read_voltage[-1]**2)/2)
            # V = self.device_read_voltage[-1] #worst case estimation
            V = math.sqrt(0.9*(self.device_read_voltage[0]**2) + 0.1*(self.device_read_voltage[-1]**2))
        assert V >= 0, "Voltage < 0"
        self.device_read_power = V ** 2 / R

    # 一个单元的写功率
    def calculate_device_write_power(self, R = None, V = None):
        # only used for NVM
        # R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
        # assert self.type == "NVM", "only the NVM device write power needs to be calculated"
        if R is None:
            R = math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
        assert R > 0, "Resistance <= 0"
        if V is None:
            V = (self.device_write_voltage[0] + self.device_write_voltage[-1])/2
        assert V >= 0, "Voltage < 0"
        self.device_write_power = V ** 2 / R

