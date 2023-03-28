import torch.nn as nn
import systemParameter as para


class latencyModule:
    def __init__(self, net: nn.Module):
        self.layer_num = 0
        for layer in net.modules():
            if hasattr(layer, 'weightBits'):
                self.layer_num += 1

        self.net = net
        # self.layer_latency_list = []
        # self.read_array_latency = 0.0
        # self.read_buffer_latency = 0.0
        #
        # self.write_array_latency = 0.0
        # self.write_buffer_latency = 0.0
        #
        # self.mm_array_latency = 0.0
        # self.mm_dac_latency = 0.0
        # self.mm_adc_latency = 0.0
        self.all_latency = 0.0
        # self.epoch_layer_latency_list = []
        self.calculate_latency()

    def calculate_latency(self):
        for epoch_index in range(para.epochs):
            for iter_index in range(para.iterations):
                self.all_latency += self.calculate_iter_latency(iter_index)

    def calculate_iter_latency(self, iter_index):
        iter_data_size = para.data_size_in_iter[iter_index]

        layer_latency_list = self.produce_layer_latency_list()

        if para.compute_mode == 'pipeline':
            max_layer_latency = max(layer_latency_list)
            self.all_latency += (len(layer_latency_list) + iter_data_size - 1) * max_layer_latency
        elif para.compute_mode == 'sequential':
            self.all_latency += sum(layer_latency_list) * iter_data_size
        else:
            raise Exception('illegal compute mode: ' + para.compute_mode)
        return iter_index

    def produce_layer_latency_list(self):
        layer_latency_list = []

        for layer in self.net.modules():
            if hasattr(layer, 'weightBits'):
                layer_latency_list.append(self.calculate_layer_latency(layer))

        layer_latency_list.append(self.calculate_loss_latency())

        if para.training:
            for layer in self.net.modules():
                if hasattr(layer, 'weightBits'):
                    layer_latency_list.append(self.calculate_layer_latency(layer, backward=True))

        return layer_latency_list

    def calculate_layer_latency(self, layer, backward=False):
        if backward:
            # backward latency:
            # Todo:
            return 1.
        else:
            # forward latency:
            # Todo:
            return 2.

    def calculate_loss_latency(self):
        # todo: to be completed
        loss_latency = 0.
        loss_latency += 1
        if para.training:
            loss_latency += 1

        return loss_latency

    def calculate_update_latency(self, module):
        if para.training:
            # todo:
            return 1.
        else:
            raise Exception('Inference without update latency')
