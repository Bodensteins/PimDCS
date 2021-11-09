# -*- coding: utf-8 -*-
import torch
import quantization
from pimlinear import PimLinear
from collections import defaultdict
from quantization import add_alpha_tensor_, mul_num, de_quantization, parse_quantization_para, float_to_int, \
    remove_additional_col, add_additional_col_of_zero, write_array_
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from enum import Enum


class OptimMode(Enum):
    full_fix = 0
    float_weight = 1

class PimSGD:
    def __init__(self, network, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, run_mode = OptimMode.full_fix):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        self.runMode = run_mode
        self.state = defaultdict(dict)
        self.defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                             weight_decay=weight_decay, nesterov=nesterov)
        wtArrays = []
        wArrays = []
        wArrays_configs = []
        wtArrays_configs = []
        delta_qweight_t_config = []
        weight = []
        for layer in network.named_modules():
            if isinstance(layer[1], PimLinear):
                wArrays.append(layer[1].wArr)
                wtArrays.append(layer[1].wtArr)
                wArrays_configs.append(layer[1].wArrConfig)
                wtArrays_configs.append(layer[1].wtArrConfig)
                delta_qweight_t_config.append(layer[1].delta_qweight_t_config)
                weight.append(layer[1].weight)
        self.param_group = {'wArr': wArrays, 'wtArr': wtArrays,
                            'wArr_configs': wArrays_configs, 'wtArr_configs': wtArrays_configs,
                            'delta_qweight_t_config': delta_qweight_t_config,
                            'weight': weight}
        self.param_group.update(self.defaults)


    def zero_grad(self):
        pass


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        weight_decay = self.param_group['weight_decay']
        momentum = self.param_group['momentum']
        dampening = self.param_group['dampening']
        nesterov = self.param_group['nesterov']
        lr = self.param_group['lr']

        for wArr, wtArr, wArr_cfg, wtArr_cfg, d_wt_cfg, weight in zip(
                self.param_group['wArr'], self.param_group['wtArr'],
                self.param_group['wArr_configs'],
                self.param_group['wtArr_configs'],
                self.param_group['delta_qweight_t_config'],
                self.param_group['weight']):
            d_wt = wtArr.grad
            if d_wt is not None:
                # print(parse_quantization_para(float_to_int(d_wt_cfg)))
                # print(de_quantization([d_wt, d_wt_cfg]))
                # print(de_quantization([wtArr, wtArr_cfg]).t())

                state = self.state[wArr]

                # merge
                if weight_decay != 0:
                    add_alpha_tensor_([d_wt, d_wt_cfg], [
                                      wtArr, wtArr_cfg], weight_decay)
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = torch.clone(d_wt).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf, d_wt_cfg = mul_num([buf, d_wt_cfg], momentum)
                        add_alpha_tensor_([buf, d_wt_cfg], [
                                          d_wt, d_wt_cfg], 1 - dampening)
                    if nesterov:
                        add_alpha_tensor_([d_wt, d_wt_cfg], [
                                          buf, d_wt_cfg], momentum)
                    else:
                        d_wt = buf
                    self.state[wtArr]['momentum_buffer'] = buf
                
                if self.runMode == OptimMode.full_fix:
                    add_alpha_tensor_([wtArr, wtArr_cfg], [
                                        d_wt, d_wt_cfg], -lr)
                    d_w = add_additional_col_of_zero(
                            [remove_additional_col(d_wt).t(), d_wt_cfg])
                    
                    add_alpha_tensor_([wArr, wArr_cfg], [d_w, d_wt_cfg], -lr)
                    wtArr.grad.zero_()
                elif self.runMode == OptimMode.float_weight:
                    weight.add_(-lr*de_quantization([remove_additional_col(d_wt).t(), d_wt_cfg]))
                    temp = quantization.creat_quantization_para(bit_width=16, tensor_type=quantization.TensorType.Normal)
                    x = quantization.quantization_tensor(temp, weight)
                    write_array_([wArr, wArr_cfg], [x, temp])
                    x = quantization.quantization_tensor(temp, weight.t())
                    write_array_([wtArr, wtArr_cfg], [x, temp])
                    wtArr.grad.zero_()
                    weight.grad.zero_()

        return loss
