import torch.nn as nn
import fixedPoint.nn as fpnn
from VGG_cifar10_model import fp_vgg16


class evaluationInfo(object):
    def __init__(self, model):
        super(evaluationInfo, self).__init__()
        self.model = model

        layer_list = []
        for child in model.children():
            if isinstance(child, nn.Sequential):
                for layer in child:
                    layer_list.append(layer)
            else:
                layer_list.append(child)

        for layer in layer_list:
            if isinstance(layer, fpnn.Conv2d):
                pass

            if isinstance(layer, fpnn.Linear):
                pass


def main():
    model = fp_vgg16(
        conv_input_bit_width=16,
        conv_output_bit_width=16,
        conv_weight_bit_width=16,
        conv_grad_output_bit_width=16,
        conv_next_grad_output_bit_width=16,
        conv_compute_weight_bit_width=16,
        fc_output_bit_width=16,
        fc_weight_bit_width=16,
        fc_grad_output_bit_width=16,
        fc_compute_weight_bit_width=16
    )
    ev = evaluationInfo(model)


if __name__ == "__main__":
    main()
