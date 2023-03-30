import torch
import torch.nn as nn
import fixedPoint.nn as fpnn
import systemParameter as para
import sys
sys.path.append("..")
from VGG_cifar10_model import fp_vgg16


class latencyModule:
    def __init__(self, net: nn.Module):
        self.net = net
        self.all_latency = 0.0
        self.stage_latency_list = self.produce_stage_latency_list()
        self.normal_iter_compute_latency = self.calculate_iter_compute_latency(para.batch_size)
        self.last_iter_compute_latency = self.calculate_iter_compute_latency(para.last_iter_data_size)
        self.calculate_latency()

    def calculate_latency(self):
        for epoch_index in range(para.epochs):
            for iter_index in range(para.iterations):
                is_last_iter = (iter_index == para.iterations - 1)
                self.all_latency += self.calculate_iter_latency(epoch_index, iter_index, is_last_iter)

    def calculate_iter_latency(self, epoch_index, iter_index, is_last_iter=False):
        latency = 0.

        if is_last_iter:
            latency += self.last_iter_compute_latency
        else:
            latency += self.normal_iter_compute_latency

        if para.training:
            update_latency = self.calculate_update_latency(epoch_index, iter_index)
            latency += update_latency

        return latency

    def produce_stage_latency_list(self):
        stage_latency_list = []
        previous_stage = []  # may include multiple layers: cnn/fc + relu(op) + pool(op)...

        # forward
        for layer in self.net.modules():
            if hasattr(layer, 'weightBits'):
                stage_latency = self.calculate_stage_latency(previous_stage)

                if stage_latency != 0:  # latency = 0 means not a real stage
                    stage_latency_list.append(stage_latency)

                previous_stage.clear()

            previous_stage.append(layer)

        # process the last stage
        stage_latency = self.calculate_stage_latency(previous_stage)
        if stage_latency == 0:
            raise Exception("the last stage should not be virtual stage")
        stage_latency_list.append(stage_latency)

        # calculate loss and err
        stage_latency = self.calculate_loss_latency()
        stage_latency_list.append(stage_latency)

        # backward
        # todo: to be completed
        if para.training:
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

    def calculate_stage_latency(self, stage, backward=False):
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
            # transfer activation latency (to next layer, for backward)
            # Todo:
            for layer in stage:
                if isinstance(layer, fpnn.Linear):
                    latency += 1.
                elif isinstance(layer, fpnn.ReLU):
                    latency += 2.
                elif isinstance(layer, fpnn.Dropout):
                    latency += 3.0
                elif isinstance(layer, nn.MaxPool2d):
                    latency += 4.0
                elif isinstance(layer, fpnn.Conv2d):
                    latency += 5.0
                elif isinstance(layer, nn.ReLU):
                    latency += 6.0
                elif not isinstance(layer, nn.Flatten) \
                        and not isinstance(layer, fpnn.Quan)\
                        and not isinstance(layer, fpnn.DeQuan):
                    # don't support bn layer now
                    raise Exception('illegal layer type' + layer.__class__.__name__)

            return 2.

    def calculate_iter_compute_latency(self, iter_data_size):
        # just don't include weight update latency
        latency = 0.

        if para.compute_mode == 'pipeline':
            max_stage_latency = max(self.stage_latency_list)
            latency += (len(self.stage_latency_list) + iter_data_size - 1) * max_stage_latency
        elif para.compute_mode == 'sequential':
            latency += sum(self.stage_latency_list) * iter_data_size
        else:
            raise Exception('illegal compute mode: ' + para.compute_mode)

        return latency


    def calculate_loss_latency(self):
        # todo: to be completed
        # transfer to cpu or need hardware do this? calculate loss and error
        loss_latency = 0.
        loss_latency += 1

        if para.training:
            loss_latency += 1

        return loss_latency

    def calculate_update_latency(self, epoch_index, iter_index, other=None):
        if para.training:
            # todo:
            return 1.
        else:
            raise Exception('Inference without update latency')


def test():
    model = fp_vgg16(*para.bit_width_tuple)
    device = torch.device("cuda:2")
    model.to(device)
    lm = latencyModule(model)
    pass


if __name__ == "__main__":
    test()
