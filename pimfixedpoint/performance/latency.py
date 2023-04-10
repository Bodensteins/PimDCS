import math

import torch
import torch.nn as nn
import fixedPoint.nn as fpnn
import systemParameter as sysPara
import sys
sys.path.append("..")
from VGG_cifar10_model import fp_vgg16
from utils import int_div_ceil


class latencyModule:
    def __init__(self, net: nn.Module):
        self.net = net
        self.epochs = sysPara.epochs
        self.iterations = sysPara.iterations
        self.training = sysPara.training
        self.compute_mode = sysPara.compute_mode

        self.input_data_shape = sysPara.input_data_shape
        self.layer_type_list = []
        self.data_shape_list = []

        self.analyze_network()

        print("layers: ", self.layer_type_list)
        # print(len(self.layer_type_list))
        # print(self.data_shape_list)
        # print(len(self.data_shape_list))

        self.activation_unit_num = sysPara.activation_unit_num
        self.activation_unit_latency = sysPara.activation_unit_latency

        self.all_latency = 0.0
        self.stage_latency_list = self.produce_stage_latency_list()
        self.normal_iter_compute_latency = self.calculate_iter_compute_latency(sysPara.batch_size)
        self.last_iter_compute_latency = self.calculate_iter_compute_latency(sysPara.last_iter_data_size)
        self.calculate_latency()

    def analyze_network(self):
        data_shape = self.input_data_shape
        layer_type_list = []
        data_shape_list = [data_shape]

        for name, module in self.net.named_modules():
            children_num = sum(1 for _ in module.children())
            if children_num == 0:  # leaf node
                if isinstance(module, fpnn.Linear):
                    layer_type_list.append("Linear")
                    if len(data_shape) != 1 or data_shape[0] != module.in_features:
                        raise Exception("illegal shape, input data shape: " + str(data_shape) + ", but in features: " + str(module.in_features))
                    data_shape = (module.out_features,)
                    data_shape_list.append(data_shape)
                elif isinstance(module, fpnn.ReLU) or isinstance(module, nn.ReLU):
                    layer_type_list.append("ReLU")
                    data_shape_list.append(data_shape)
                elif isinstance(module, fpnn.Dropout):
                    layer_type_list.append("Dropout")
                    data_shape_list.append(data_shape)
                elif isinstance(module, nn.MaxPool2d):
                    layer_type_list.append("MaxPool2d")
                    data_shape = (data_shape[0] // 2, data_shape[1] // 2, data_shape[2])
                    data_shape_list.append(data_shape)
                elif isinstance(module, fpnn.Conv2d):
                    layer_type_list.append("Conv2d")
                    data_shape = (data_shape[0], data_shape[1], module.out_channels)
                    data_shape_list.append(data_shape)
                elif isinstance(module, nn.Flatten):
                    data_shape = (math.prod(data_shape),)
                    # layer_type_list.append("Flatten")
                    pass
                elif not isinstance(module, fpnn.Quan) and not isinstance(module, fpnn.DeQuan):
                    raise Exception("illegal module: " + name)

        self.layer_type_list = layer_type_list
        self.data_shape_list = data_shape_list

    def calculate_latency(self):
        for epoch_index in range(self.epochs):
            for iter_index in range(self.iterations):
                is_last_iter = (iter_index == self.iterations - 1)
                self.all_latency += self.calculate_iter_latency(epoch_index, iter_index, is_last_iter)

    def calculate_iter_latency(self, epoch_index, iter_index, is_last_iter=False):
        latency = 0.

        if is_last_iter:
            latency += self.last_iter_compute_latency
        else:
            latency += self.normal_iter_compute_latency

        if self.training:
            update_latency = self.calculate_update_latency(epoch_index, iter_index)
            latency += update_latency

        return latency

    def produce_stage_latency_list(self):
        stage_latency_list = []
        previous_stage = []  # may include multiple layers: cnn/fc + relu(op) + pool(op)...
        start_layer_index = 0
        end_layer_index = 0

        # forward
        for index, name in enumerate(self.layer_type_list):
            if (name == 'Conv2d' or name == 'Linear') and previous_stage:
                stage_latency = self.calculate_stage_latency(previous_stage, start_layer_index=start_layer_index)
                start_layer_index = end_layer_index
                if stage_latency != 0:  # latency = 0 means not a real stage
                    stage_latency_list.append(stage_latency)

                previous_stage.clear()

            previous_stage.append(name)
            end_layer_index += 1
        # for layer in self.net.modules():
        #     if hasattr(layer, 'weightBits'):
        #         stage_latency = self.calculate_stage_latency(previous_stage)
        #
        #         if stage_latency != 0:  # latency = 0 means not a real stage
        #             stage_latency_list.append(stage_latency)
        #
        #         previous_stage.clear()
        #
        #     previous_stage.append(layer)

        # process the last stage
        stage_latency = self.calculate_stage_latency(previous_stage, start_layer_index)
        if stage_latency == 0:
            raise Exception("the last stage should not be virtual stage")
        stage_latency_list.append(stage_latency)

        # calculate loss and err
        stage_latency = self.calculate_loss_latency()
        stage_latency_list.append(stage_latency)

        # backward
        # todo: to be completed
        if self.training:
            previous_stage = []
            for layer in self.net.modules():
                pass
                # if hasattr(layer, 'weightBits'):
                #     if isinstance(next(layer), nn.ReLU):
                #         pass
                # else:
                #     previous_layers.append(layer)
                # stage_latency_list.append(self.calculate_stage_latency(layer, backward=True))

        return stage_latency_list

    def calculate_stage_latency(self, stage, start_layer_index=0, backward=False):
        # print("stage: ", stage)
        # print("start_layer_index: " + str(start_layer_index))
        latency = 0.

        if backward:
            # backward latency:
            # Todo:
            return 1.
        else:
            # forward latency:
            # read data from array level buffer
            # dac latency
            # mvm latency
            # s&h latency
            # adc latency
            # shifter and adder latency for different bit slice(include different input and different weight)
            # relu latency(op)
            # pool latency(op)
            # transfer activation latency (to next layer_name, for backward)
            # Todo:
            input_element_number = math.prod(self.data_shape_list[start_layer_index])
            for layer_name in stage:
                if layer_name == "Linear":
                    latency += 1.
                elif layer_name == "ReLU":
                    latency += self.calculate_activation_unit_latency(input_element_number)
                elif layer_name == "Dropout":
                    latency += 3.0
                elif layer_name == "MaxPool2d":
                    latency += 4.0
                elif layer_name == "Conv2d":
                    latency += 5.0
                else:
                    # don't support bn layer_name now
                    raise Exception('illegal layer_name type: ' + layer_name)

            return 2.

    def calculate_iter_compute_latency(self, iter_data_size):
        # just don't include weight update latency
        latency = 0.

        if self.compute_mode == 'pipeline':
            max_stage_latency = max(self.stage_latency_list)
            latency += (len(self.stage_latency_list) + iter_data_size - 1) * max_stage_latency
        elif self.compute_mode == 'sequential':
            latency += sum(self.stage_latency_list) * iter_data_size
        else:
            raise Exception('illegal compute mode: ' + self.compute_mode)

        return latency

    def calculate_loss_latency(self):
        # todo: to be completed
        # transfer to cpu or need hardware do this? calculate loss and error
        loss_latency = 0.
        loss_latency += 1

        if self.training:
            loss_latency += 1

        return loss_latency

    def calculate_update_latency(self, epoch_index, iter_index, other=None):
        if self.training:
            # todo:
            return 1.
        else:
            raise Exception('Inference without update latency')

    def calculate_activation_unit_latency(self, element_num=0):
        calculate_times = int_div_ceil(element_num, self.activation_unit_num)

        return calculate_times * self.activation_unit_latency


def test():
    model = fp_vgg16(*sysPara.bit_width_tuple)
    device = torch.device("cuda:2")
    model.to(device)
    lm = latencyModule(model)
    pass


if __name__ == "__main__":
    test()
