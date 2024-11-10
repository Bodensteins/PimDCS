# from MNSIM

import configparser as cp

class JointModule:
    def __init__(self, SimConfig_path, max_bitwidth = None):
        # frequency unit: MHz
        jointmodule_config = cp.ConfigParser()
        jointmodule_config.read(SimConfig_path, encoding='UTF-8')
        self.jointmodule_tech = int(jointmodule_config.get('Digital module', 'JointModule_Tech'))
        if self.jointmodule_tech <= 0:
            self.jointmodule_tech = 65
        self.jointmodule_area = float(jointmodule_config.get('Digital module', 'JointModule_Area'))
        self.jointmodule_power = float(jointmodule_config.get('Digital module', 'JointModule_Power'))
        if max_bitwidth is None:
            self.jointmodule_bit = 8
        else:
            self.jointmodule_bit = max_bitwidth
        assert self.jointmodule_bit > 0
        self.jointmodule_frequency = float(jointmodule_config.get('Digital module', 'Digital_Frequency'))
        if self.jointmodule_frequency == 0:
            self.jointmodule_frequency = 100
        assert self.jointmodule_frequency > 0
        self.jointmodule_latency = 1.0 / self.jointmodule_frequency
        self.adder_energy = 0
        self.calculate_jointmodule_power()

    def calculate_jointmodule_power(self):
        # unit: W
        if self.jointmodule_power == 0:
            jointmodule_power_dict = {4: 1.39e-4,
                               8: 2.64e-4,
                               12: 3.67e-4,
                               16: 4.97e-4
                               }
            if self.jointmodule_bit <= 4:
                self.jointmodule_power = jointmodule_power_dict[4]*pow((self.jointmodule_tech/65),2)
            elif self.jointmodule_bit <= 8:
                self.jointmodule_power = jointmodule_power_dict[8] * pow((self.jointmodule_tech / 65), 2)
            elif self.jointmodule_bit <= 12:
                self.jointmodule_power = jointmodule_power_dict[12] * pow((self.jointmodule_tech / 65), 2)
            else:
                self.jointmodule_power = jointmodule_power_dict[16] * pow((self.jointmodule_tech / 65), 2)

    def calculate_jointmodule_area(self):
        # unit: um^2
        if self.jointmodule_area == 0:
            jointmodule_area_dict = {4: 182.88,
                               8: 353.76,
                               12: 385.44,
                               16: 512.16
                               }
            if self.jointmodule_bit <= 4:
                self.jointmodule_area = jointmodule_area_dict[4]*pow((self.jointmodule_tech/65),2)
            elif self.jointmodule_bit <= 8:
                self.jointmodule_area = jointmodule_area_dict[8] * pow((self.jointmodule_tech / 65), 2)
            elif self.jointmodule_bit <= 12:
                self.jointmodule_area = jointmodule_area_dict[12] * pow((self.jointmodule_tech / 65), 2)
            else:
                self.jointmodule_area = jointmodule_area_dict[16] * pow((self.jointmodule_tech / 65), 2)

    def calculate_jointmodule_energy(self):
        assert self.jointmodule_power >= 0
        assert self.jointmodule_latency >= 0
        self.jointmodule_energy = self.jointmodule_latency * self.jointmodule_power