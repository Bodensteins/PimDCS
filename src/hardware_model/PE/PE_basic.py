import configparser as cp
import math
import os
from src.hardware_model.basic_component.ADC import ADC
from src.hardware_model.basic_component.DAC import DAC
from src.hardware_model.basic_component.adder import Adder
from src.hardware_model.basic_component.buffer import Buffer
from src.hardware_model.basic_component.reg import Reg
from src.hardware_model.basic_component.shiftreg import ShiftReg
from src.hardware_model.basic_component.mux import Mux
from src.hardware_model.basic_component.demux import Demux
from src.hardware_model.xbar.xbar_basic import Crossbar_Basic
from src.hardware_model.adder_tree import AdderTree_PE
from src.hardware_model.accumulator import Accumulator_PE

# OU计算
# 1T1R，分正负阵列，模拟域相减（电压差）
# SH缓存电压差？
class ProcessElement_Basic:
    def __init__(self, SimConfig_path):
        PE_config = cp.ConfigParser()
        PE_config.read(SimConfig_path, encoding='UTF-8')

        self.input_bitwidth = 8 # temp
        self.xbar_group_num = int(PE_config.get('Process element level', 'Group_Num'))
        self.xbar_ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))
        self.xbar_DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))
        self.weight_bitwidth = self.xbar_group_num

        # 缺少一个input reg mux
        self.input_buffer_model = Buffer(SimConfig_path, buf_level=1, buf_size=1)  # buf_level和buf_size修改
        self.input_reg_model = Reg(SimConfig_path, bitwidth=self.input_bitwidth)
        self.DAC_model = DAC(SimConfig_path)
        self.xbar_model = Crossbar_Basic(SimConfig_path)
        self.ADC_model = ADC(SimConfig_path)
        # self.adder_model = Adder(SimConfig_path)
        self.shiftreg_model = ShiftReg(SimConfig_path, max_shiftbase=self.weight_bitwidth + self.input_bitwidth)
        self.addertree_model = AdderTree_PE(SimConfig_path)
        self.output_reg_model = Reg(SimConfig_path, bitwidth=16)
        self.accumulator_model = Accumulator_PE(SimConfig_path)
        self.ADC_mux_model = Mux(SimConfig_path, mux_in_num=math.ceil(self.xbar_model.ou_column / self.xbar_ADC_num))
        self.ADC_demux_model = Demux(SimConfig_path, demux_out_num=math.ceil(self.xbar_model.ou_column / self.xbar_ADC_num))
        self.accumulator_demux_model = Demux(SimConfig_path, demux_out_num=self.xbar_model.xbar_column // self.xbar_model.ou_column)
        # input_reg缺少mux
        # accumulator到output reg缺少mux
        # 输出双缓冲需要额外的mux和demux
        
        self.cell_type = self.xbar_model.cell_type
        self.PE_xbar_num = self.xbar_group_num * 2 # 正负阵列

        # print(self.xbar_ADC_num)
        self.PE_ADC_num = self.xbar_ADC_num * self.xbar_group_num
        self.PE_DAC_num = self.xbar_DAC_num * self.xbar_group_num
        self.PE_input_reg_num = self.xbar_model.xbar_row
        # self.PE_adder_num = None
        self.PE_shiftreg_num = self.xbar_model.ou_column
        self.PE_ADC_mux_num = self.xbar_ADC_num
        self.PE_ADC_demux_num = self.xbar_ADC_num
        self.PE_accumulator_demux_num = self.accumulator_model.adder_num
        self.PE_output_reg_num = self.xbar_model.xbar_column * 2    #输出双缓冲
        # 输出双缓冲需要额外的mux和demux
    
        self.mutiple_time = self.xbar_model.ou_num * math.ceil(self.input_bitwidth / self.DAC_model.DAC_precision)

        # area
        self.PE_input_buffer_area = 0
        self.PE_input_reg_area = 0
        self.PE_xbar_area = 0
        self.PE_ADC_area = 0
        self.PE_ADC_demux_area = 0
        self.PE_ADC_mux_area = 0
        self.PE_DAC_area = 0
        self.PE_shiftreg_area = 0
        self.PE_addertree_area = 0
        self.PE_accumulator_demux_area = 0
        self.PE_accumulator_area = 0
        self.PE_output_reg_area = 0
        self.PE_digital_area = 0
        self.PE_analog_area = 0
        self.PE_area = 0

        # latency
        self.PE_buffer_latency = 0
        self.PE_input_buffer_rlatency = 0
        self.PE_input_buffer_wlatency = 0
        self.PE_input_reg_rlatency = 0
        self.PE_input_reg_wlatency = 0
        self.PE_xbar_latency = 0
        self.PE_ADC_latency = 0
        self.PE_ADC_demux_latency = 0
        self.PE_ADC_mux_latency = 0
        self.PE_DAC_latency = 0
        self.PE_shiftreg_rlatency = 0
        self.PE_shiftreg_wlatency = 0
        self.PE_addertree_latency = 0
        self.PE_accumulator_demux_latency = 0
        self.PE_accumulator_latency = 0
        self.PE_output_reg_rlatency = 0
        self.PE_output_reg_wlatency = 0
        self.PE_digital_latency = 0
        self.PE_analog_latency = 0
        self.PE_latency = 0

        # energy
        self.PE_buffer_energy = 0
        self.PE_input_buffer_renergy = 0
        self.PE_input_buffer_wenergy = 0
        self.PE_input_reg_renergy = 0
        self.PE_input_reg_wenergy = 0
        self.PE_xbar_energy = 0
        self.PE_ADC_energy = 0
        self.PE_ADC_demux_energy = 0
        self.PE_ADC_mux_energy = 0
        self.PE_DAC_energy = 0
        self.PE_shiftreg_renergy = 0
        self.PE_shiftreg_wenergy = 0
        self.PE_addertree_energy = 0
        self.PE_accumulator_demux_energy = 0
        self.PE_accumulator_energy = 0
        self.PE_output_reg_renergy = 0
        self.PE_output_reg_wenergy = 0
        self.PE_digital_energy = 0
        self.PE_analog_energy = 0
        self.PE_energy = 0

        # power
        self.PE_buffer_power = 0
        self.PE_analog_power = 0
        self.PE_digital_power = 0
        self.PE_power = 0

    def PE_read_config(self):
        pass
    
    def calculate_PE_area(self):
        self.input_buffer_model.calculate_buf_area()
        self.input_reg_model.calculate_reg_area()
        self.DAC_model.calculate_DAC_area()
        self.xbar_model.calculate_xbar_area()
        self.ADC_model.calculate_ADC_area()
        self.ADC_mux_model.calculate_mux_area()
        self.ADC_demux_model.calculate_demux_area()
        self.shiftreg_model.calculate_shiftreg_area()
        self.addertree_model.calculate_addertree_area()
        self.accumulator_model.calculate_accumulator_area()
        self.accumulator_demux_model.calculate_demux_area()
        self.output_reg_model.calculate_reg_area()

        self.PE_input_buffer_area = self.input_buffer_model.buf_area
        self.PE_input_reg_area = self.input_reg_model.reg_area * self.PE_input_reg_num
        self.PE_xbar_area = self.xbar_model.xbar_area * self.PE_xbar_num
        self.PE_DAC_area = self.DAC_model.DAC_area * self.PE_DAC_num
        self.PE_ADC_area = self.ADC_model.ADC_area * self.PE_ADC_num
        self.PE_ADC_mux_area = self.ADC_mux_model.mux_area * self.PE_ADC_mux_num
        self.PE_ADC_demux_area = self.ADC_demux_model.demux_area * self.PE_ADC_demux_num
        self.PE_shiftreg_area = self.shiftreg_model.shiftreg_area * self.PE_shiftreg_num
        self.PE_addertree_area = self.addertree_model.addertree_area
        self.PE_accumulator_area = self.accumulator_model.accumulator_area
        self.PE_accumulator_demux_area = self.accumulator_demux_model.demux_area * self.PE_accumulator_demux_num
        self.PE_output_reg_area = self.output_reg_model.reg_area * self.PE_output_reg_num

        self.PE_digital_area = self.PE_input_reg_area + self.PE_ADC_demux_area + self.PE_shiftreg_area + \
            self.PE_addertree_area + self.PE_accumulator_area + self.PE_accumulator_demux_area + self.PE_output_reg_area
        self.PE_analog_area = self.PE_DAC_area + self.PE_ADC_area + self.PE_ADC_mux_area
        self.PE_area = self.PE_input_buffer_area + self.PE_analog_area + self.PE_digital_area


    def calculate_PE_latency_no_pipeline(self):
        self.input_buffer_model.calculate_buf_write_latency(wdata=1)     # wdata未确定
        self.input_buffer_model.calculate_buf_read_latency(rdata=1)     # rdata未确定
        # self.input_buffer_model.calculate_buf_latency()
        # self.input_reg_model.calculate_reg_latency()
        self.DAC_model.calculate_DAC_latency()
        self.xbar_model.calculate_xbar_read_latency()
        self.ADC_model.calculate_ADC_latency()
        self.ADC_mux_model.calculate_mux_latency()
        self.ADC_demux_model.calculate_demux_latency()
        # self.shiftreg_model.calculate_shiftreg_latency()
        self.addertree_model.calculate_addertree_latency()
        self.accumulator_model.calculate_accumulator_latency()
        self.accumulator_demux_model.calculate_demux_latency()
        # self.output_reg_model.calculate_reg_latency()

        self.PE_input_buffer_wlatency = self.input_buffer_model.buf_wlatency
        self.PE_input_buffer_rlatency = self.input_buffer_model.buf_rlatency

        self.PE_input_reg_wlatency = self.input_reg_model.reg_latency * math.ceil(self.input_bitwidth / self.DAC_model.DAC_precision)
        self.PE_input_reg_rlatency = self.input_reg_model.reg_latency * self.mutiple_time   # 包括读和写两部分

        self.PE_DAC_latency = self.DAC_model.DAC_latency * self.mutiple_time
        self.PE_xbar_latency = self.xbar_model.xbar_read_latency * self.mutiple_time
        self.PE_ADC_mux_latency = self.ADC_mux_model.mux_latency * self.ADC_mux_model.mux_in_num * self.mutiple_time
        self.PE_ADC_latency = self.ADC_model.ADC_latency * math.ceil(self.xbar_model.ou_column / self.xbar_ADC_num) * self.mutiple_time
        self.PE_ADC_demux_latency = self.ADC_demux_model.demux_latency * self.ADC_demux_model.demux_out_num * self.mutiple_time
        self.PE_shiftreg_wlatency = self.shiftreg_model.shiftreg_latency * self.mutiple_time
        self.PE_shiftreg_rlatency = self.shiftreg_model.shiftreg_latency * self.mutiple_time     # 包括读和写两部分

        self.PE_addertree_latency = self.addertree_model.addertree_latency * self.mutiple_time
        self.PE_accumulator_latency = self.accumulator_model.accumulator_latency * self.mutiple_time
        self.PE_accumulator_demux_latency = self.accumulator_demux_model.demux_latency * self.mutiple_time  # 包括读和写两部分
        self.PE_output_reg_rlatency = self.output_reg_model.reg_latency * self.mutiple_time
        self.PE_output_reg_wlatency = self.output_reg_model.reg_latency * self.mutiple_time

        self.PE_buffer_latency = self.PE_input_buffer_rlatency + self.PE_input_buffer_wlatency
        self.PE_digital_latency = self.PE_input_reg_wlatency + self.PE_input_reg_rlatency + self.PE_ADC_demux_latency + self.PE_shiftreg_rlatency + self.PE_shiftreg_wlatency + \
            self.PE_addertree_latency + self.PE_accumulator_latency + self.PE_accumulator_demux_latency + self.PE_output_reg_wlatency + self.PE_output_reg_rlatency
        self.PE_analog_latency = self.PE_DAC_latency + self.PE_ADC_latency + self.PE_ADC_mux_latency
        self.PE_latency = self.PE_buffer_latency + self.PE_analog_latency + self.PE_digital_latency

    def calculate_PE_latency_pipeline(self, read_row=None, read_column=None):
        pass

    def calculate_PE_energy(self):
        self.input_buffer_model.calculate_buf_write_energy(wdata=1)     # wdata未确定
        self.input_buffer_model.calculate_buf_read_energy(rdata=1)     # rdata未确定
        self.PE_input_buffer_wenergy = self.input_buffer_model.buf_wenergy
        self.PE_input_buffer_renergy = self.input_buffer_model.buf_renergy

        self.input_reg_model.calculate_reg_energy()
        self.PE_input_reg_wenergy = self.input_reg_model.reg_energy * self.PE_input_reg_num * math.ceil(self.input_bitwidth / self.DAC_model.DAC_precision)
        self.PE_input_reg_renergy = self.input_reg_model.reg_energy * self.xbar_model.ou_row * self.mutiple_time   # 包括读和写两部分，每个ou计算只激活ou_row_num次input_reg

        self.DAC_model.calculate_DAC_energy()
        self.PE_DAC_energy = self.DAC_model.DAC_energy * self.xbar_model.ou_row * self.xbar_group_num * self.mutiple_time   # 每个ou计算只激活ou_row_num次DAC

        self.xbar_model.calculate_xbar_read_energy()
        self.PE_xbar_energy = self.xbar_model.xbar_read_energy * self.xbar_group_num * self.mutiple_time

        self.ADC_mux_model.calculate_mux_energy()
        self.PE_ADC_mux_energy = self.ADC_mux_model.mux_energy * self.xbar_model.ou_column * self.xbar_group_num * self.mutiple_time

        self.ADC_model.calculate_ADC_energy()
        self.PE_ADC_energy = self.ADC_model.ADC_energy * self.xbar_model.ou_column * self.xbar_group_num * self.mutiple_time

        self.ADC_demux_model.calculate_demux_energy()
        self.PE_ADC_demux_energy = self.ADC_demux_model.demux_energy * self.xbar_model.ou_column * self.xbar_group_num * self.mutiple_time

        self.shiftreg_model.calculate_shiftreg_energy()
        self.PE_shiftreg_wenergy = self.shiftreg_model.shiftreg_energy * self.xbar_model.ou_column * self.xbar_group_num * self.mutiple_time
        self.PE_shiftreg_renergy = self.shiftreg_model.shiftreg_energy * self.xbar_model.ou_column * self.xbar_group_num * self.mutiple_time

        self.addertree_model.calculate_addertree_energy()
        self.PE_addertree_energy = self.addertree_model.addertree_energy * self.mutiple_time

        self.accumulator_model.calculate_accumulator_energy()
        self.PE_accumulator_energy = self.accumulator_model.accumulator_energy * self.mutiple_time
        
        self.accumulator_demux_model.calculate_demux_energy()
        self.PE_accumulator_demux_energy = self.accumulator_demux_model.demux_energy * self.PE_accumulator_demux_num * self.mutiple_time

        self.output_reg_model.calculate_reg_energy()
        self.PE_output_reg_renergy = self.output_reg_model.reg_energy * self.accumulator_model.adder_num * self.mutiple_time
        self.PE_output_reg_wenergy = self.output_reg_model.reg_energy * self.accumulator_model.adder_num * self.mutiple_time

        self.PE_buffer_energy = self.PE_input_buffer_renergy + self.PE_input_buffer_wenergy
        self.PE_analog_energy = self.PE_DAC_energy + self.PE_xbar_energy + self.PE_ADC_mux_energy + self.PE_ADC_energy
        self.PE_digital_energy = self.PE_ADC_demux_energy + self.PE_shiftreg_wenergy + self.PE_shiftreg_renergy + self.PE_addertree_energy + \
            self.PE_accumulator_energy + self.PE_accumulator_demux_energy + self.PE_output_reg_renergy + self.PE_output_reg_wenergy
        self.PE_energy = self.PE_buffer_energy + self.PE_analog_energy + self.PE_digital_energy

    def calculate_PE_power_no_pipeline(self):
        self.PE_buffer_power = (self.PE_input_buffer_renergy + self.PE_input_buffer_wenergy) / (self.PE_input_buffer_rlatency + self.PE_input_buffer_wlatency)
        self.PE_analog_power = self.PE_analog_energy / self.PE_analog_latency
        self.PE_digital_power = self.PE_digital_energy / self.PE_digital_latency
        self.PE_power = self.PE_energy / self.PE_latency

    def calculate_PE_power_pipeline(self):
        pass

    def PE_print_metrics(self):
        # print("---------------------Crossbar Configurations-----------------------")
        # crossbar.xbar_output(self)
        # print("------------------------DAC Configurations-------------------------")
        # DAC.DAC_output(self)
        # print("------------------------ADC Configurations-------------------------")
        # ADC.ADC_output(self)
        print("-------------------------PE Configurations-------------------------")
        print("total crossbar number in one PE:", self.PE_xbar_num)
        # print("			the number of crossbars sharing a set of interfaces:",self.PE_multiplex_xbar_num)
        # print("total utilization rate:", self.PE_utilization)
        print("total DAC number in one PE:", self.PE_DAC_num)
        print("			the number of DAC in one set of interfaces:", self.xbar_DAC_num)
        print("total ADC number in one PE:", self.PE_ADC_num)
        print("			the number of ADC in one set of interfaces:", self.xbar_ADC_num)
        print("---------------------PE Area Simulation Results--------------------")
        print("PE area:", self.PE_area, "um^2")
        print("			input buffer area:", self.PE_input_buffer_area, "um^2")
        print("			crossbar area:", self.PE_xbar_area, "um^2")
        print("			DAC area:", self.PE_DAC_area, "um^2")
        print("			ADC area:", self.PE_ADC_area, "um^2")
        print("			digital part area:", self.PE_digital_area, "um^2")
        print("			|---adder tree area:", self.PE_addertree_area, "um^2")
        print("			|---shift-reg area:", self.PE_shiftreg_area, "um^2")
        print("			|---accumulator area:", self.PE_accumulator_area, "um^2")
        print("			|---ADC demux area:", self.PE_ADC_demux_area, "um^2")
        print("			|---ADC mux area:", self.PE_ADC_mux_area, "um^2")
        print("			|---accumulator demux area:", self.PE_accumulator_demux_area, "um^2")
        print("			|---input reg area:", self.PE_input_reg_area, "um^2")
        print("			|---output reg area:", self.PE_output_reg_area, "um^2")
        print("--------------------PE Latency Simulation Results-----------------")
        print("PE latency:", self.PE_latency, "ns")
        print("			input buffer latency:", self.PE_buffer_latency, "ns")
        print("			|---input buffer write latency:", self.PE_input_buffer_wlatency, "ns")
        print("			|---input buffer read latency:", self.PE_input_buffer_rlatency, "ns")
        print("			crossbar latency:", self.PE_xbar_latency, "ns")
        print("			DAC latency:", self.PE_DAC_latency, "ns")
        print("			ADC latency:", self.PE_ADC_latency, "ns")
        print("			digital part latency:", self.PE_digital_latency, "ns")
        print("			|---adder tree latency:", self.PE_addertree_latency, "ns")
        print("			|---shift-reg read latency:", self.PE_shiftreg_rlatency, "ns")
        print("			|---shift-reg write latency:", self.PE_shiftreg_wlatency, "ns")
        print("			|---accumulator latency:", self.PE_accumulator_latency, "ns")
        print("			|---ADC demux latency:", self.PE_ADC_demux_latency, "ns")
        print("			|---ADC mux latency:", self.PE_ADC_mux_latency, "ns")
        print("			|---accumulator demux latency:", self.PE_accumulator_demux_latency, "ns")
        print("			|---input reg read latency:", self.PE_input_reg_rlatency, "ns")
        print("			|---input reg write latency:", self.PE_input_reg_wlatency, "ns")
        print("			|---output reg read latency:", self.PE_output_reg_rlatency, "ns")
        print("			|---output reg write latency:", self.PE_output_reg_wlatency, "ns")
        print("------------------PE Energy Simulation Results--------------------")
        print("PE energy:", self.PE_energy, "nJ")
        print("			input buffer energy:", self.PE_buffer_energy, "nJ")
        print("			|---input buffer write energy:", self.PE_input_buffer_wenergy, "nJ")
        print("			|---input buffer read energy:", self.PE_input_buffer_renergy, "nJ")
        print("			crossbar energy:", self.PE_xbar_energy, "nJ")
        print("			DAC energy:", self.PE_DAC_energy, "nJ")
        print("			ADC energy:", self.PE_ADC_energy, "nJ")
        print("			digital part energy:", self.PE_digital_energy, "nJ")
        print("			|---adder tree energy:", self.PE_addertree_energy, "nJ")
        print("			|---shift-reg read energy:", self.PE_shiftreg_renergy, "nJ")
        print("			|---shift-reg write energy:", self.PE_shiftreg_wenergy, "nJ")
        print("			|---accumulator energy:", self.PE_accumulator_energy, "nJ")
        print("			|---ADC demux energy:", self.PE_ADC_demux_energy, "nJ")
        print("			|---ADC mux energy:", self.PE_ADC_mux_energy, "nJ")
        print("			|---accumulator demux energy:", self.PE_accumulator_demux_energy, "nJ")
        print("			|---input reg read energy:", self.PE_input_reg_renergy, "nJ")
        print("			|---input reg write energy:", self.PE_input_reg_wenergy, "nJ")
        print("			|---output reg read energy:", self.PE_output_reg_renergy, "nJ")
        print("			|---output reg write energy:", self.PE_output_reg_wenergy, "nJ")
        print("--------------------PE Power Simulation Results-------------------")
        print("PE power:", self.PE_power, "W")
        print("			buffer power:", self.PE_buffer_power, "W")
        print("			analog part power:", self.PE_analog_power, "W")
        print("			digital part power:", self.PE_digital_power, "W")
        

def test_PE_basic():
    test_SimConfig_path = os.path.join("/home/leitaoming/project/FADESim/", "src/hardware_model/config/hardware_config.ini")
    print("load file:", test_SimConfig_path)
    pe = ProcessElement_Basic(test_SimConfig_path)
    pe.calculate_PE_area()
    pe.calculate_PE_latency_no_pipeline()
    pe.calculate_PE_energy()
    pe.calculate_PE_power_no_pipeline()

    pe.PE_print_metrics()