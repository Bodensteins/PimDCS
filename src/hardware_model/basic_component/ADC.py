# from MNSIM


import configparser as cp


class ADC:
    def __init__(self, SimConfig_path):
        ADC_config = cp.ConfigParser()
        ADC_config.read(SimConfig_path, encoding='UTF-8')
        self.PIM_type_adc = int(ADC_config.get('Process element level', 'PIM_Type'))
        self.ADC_choice = int(ADC_config.get('Interface level', 'ADC_Choice'))
        if self.PIM_type_adc == 1 and self.ADC_choice != -1:
            # digital PIM architecture
            self.ADC_choice = 8
        self.ADC_area = float(ADC_config.get('Interface level', 'ADC_Area'))
        self.ADC_precision = int(ADC_config.get('Interface level', 'ADC_Precision'))
        self.ADC_power = float(ADC_config.get('Interface level', 'ADC_Power'))
        self.ADC_sample_rate = float(ADC_config.get('Interface level', 'ADC_Sample_Rate'))
        self.ADC_latency = 0
        self.ADC_energy = 0
        # self.ADC_interval = list(map(int, ADC_config.get('Interface level', 'ADC_Interval_Thres').split(',')))
        # print("ADC configuration is loaded")
        # self.logic_op = int(ADC_config.get('Interface level', 'Logic_Op'))
        # self.logic_op = -1
        self.calculate_ADC_precision()
        self.calculate_ADC_sample_rate()
        self.calculate_ADC_power()

    def calculate_ADC_area(self):
        #unit: um^2
        ADC_area_dict = {1: 1600, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
                         2: 1200, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
                         3: 1650, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
                         4: 580, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
                         5: 1650, #ASPDAC1
                         6: 1650, #ASPDAC2
                         7: 500, #ASPDAC3
                         8: 1, #SA @ 28nm
                         9: 15899 #Qi Liu
        }
        if self.ADC_choice != -1:
            assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
            self.ADC_area = ADC_area_dict[self.ADC_choice]
        # if self.logic_op == 0: # Notice: scale from 65nm to 28nm
        #     self.ADC_area += 17.28*0.18 # AND gate
        # elif self.logic_op == 1:
        #     self.ADC_area += 17.28*0.18 # OR gate
        # elif self.logic_op == 2:
        #     self.ADC_area += 19.2*0.18 # XOR gate

    def calculate_ADC_precision(self):
        ADC_precision_dict = {1: 10, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
                         2: 8, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
                         3: 8, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
                         4: 6, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
                         5: 8, #ASPDAC1
                         6: 6, #ASPDAC2
                         7: 4, #ASPDAC3
                         8: 1, #SA
                         9: 8
        }
        if self.ADC_choice != -1:
            assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
            self.ADC_precision = ADC_precision_dict[self.ADC_choice]

    # ADC平均功耗
    def calculate_ADC_power(self):
        #unit: W
        ADC_power_dict = {1: 6.92*1e-3, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
                         2: 2*1e-3, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
                         3: 4*1e-3, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
                         4: 1.26*1e-3, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
                         5: 4e-3, #ASPDAC1
                         6: 1.26e-3, #ASPDAC2
                         7: 0.7e-3, #ASPDAC3
                         8: 0.1086*15e-6, #SA reference: Comparative Study of Sense Amplifiers for SRAM (scale from 1.2V@65nm to 0.8V@28nm)
                         9: 8*0.0073*1e-3
        }
        if self.ADC_choice != -1:
            assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
            self.ADC_power = ADC_power_dict[self.ADC_choice]

    # 采样频率
    def calculate_ADC_sample_rate(self):
        #unit: GSamples/s
        ADC_sample_rate_dict = {1: 1.5, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
                                2: 1.28, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
                                3: 1.1, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
                                4: 1, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
                                5: 1.1, #ASPDAC1
                                6: 1, #ASPDAC2
                                7: 1,
                                8: 1, #SA reference: Comparative Study of Sense Amplifiers for SRAM
                                9: 6
        }
        if self.ADC_choice != -1:
            assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
            self.ADC_sample_rate = ADC_sample_rate_dict[self.ADC_choice]

    # 一次采样（转换）的延迟
    def calculate_ADC_latency(self):
        # unit: ns
        self.ADC_latency = 1 / self.ADC_sample_rate

    # 一次采样的能耗
    def calculate_ADC_energy(self):
        #unit: nJ
        self.ADC_energy = self.ADC_latency * self.ADC_power