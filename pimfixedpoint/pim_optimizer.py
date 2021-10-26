# -*- coding: utf-8 -*-
import torch
from pimlinear import PimLinear
from collections import defaultdict
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from torchvision.datasets import coco


class PimSGD:
    def __init__(self, network, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
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

        self.state = defaultdict(dict)
        self.defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                             weight_decay=weight_decay, nesterov=nesterov)
        wtArrays = []
        wArrays = []
        wArrays_configs = []
        wtArrays_configs = []
        delta_weight_configs = []
        for layer in network.named_modules():
            if isinstance(layer[1], PimLinear):
                wArrays.append(layer[1].wArr)
                wtArrays.append(layer[1].wtArr)
                wArrays_configs.append(layer[1].wArrConfig)
                wtArrays_configs.append(layer[1].wtArrConfig)
                delta_weight_configs.append(layer[1].delta_qweight_config)
        self.param_group = {'wArr': wArrays, 'wtArr': wtArrays, \
            'wArr_configs': wArrays_configs, 'wtArr_configs': wtArrays_configs, \
            'delta_weight_confs': delta_weight_configs}
        self.param_group.update(self.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = self.param_group['weight_decay']
        momentum = self.param_group['momentum']
        dampening = self.param_group['dampening']
        nesterov = self.param_group['nesterov']
        lr = self.param_group['lr']

        for p in zip(self.param_group['wArr'], self.param_group['wtArr'], \
            self.param_group['wArr_configs'], self.param_group['wtArr_configs'], \
                self.param_group['delta_weight_confs']):
            if p[1].grad is not None:
                params_with_grad.append(p[1])
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        # F.sgd(params_with_grad,
        #       d_p_list,
        #       momentum_buffer_list,
        #       weight_decay=weight_decay,
        #       momentum=momentum,
        #       lr=lr,
        #       dampening=dampening,
        #       nesterov=nesterov)

        for i, param in enumerate(params_with_grad):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            param.add_(d_p, alpha=-lr)

        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer

        return loss
