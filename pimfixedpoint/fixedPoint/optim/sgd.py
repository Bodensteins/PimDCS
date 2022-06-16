# -*- coding: utf-8 -*-
from . import _functional as F
from ..nn import fixedPointArithmetic as fpA
import torch
from torch.optim.optimizer import Optimizer
from .optimizer import OptimMode


class SGD(Optimizer):
    def __init__(self, params, bit_width_list: list, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 run_mode=OptimMode.full_fix):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.runMode = run_mode
        self.bit_width_list = bit_width_list

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            fp_para, fp_para_cfg = None, None
            fp_d_p, fp_d_p_cfg = None, None

            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    if i % 2 == 0:
                        fp_para = p
                        fp_d_p = p.grad
                    else:
                        fp_para_cfg = p
                        fp_d_p_cfg = p.grad
                        params_with_grad.append((fp_para, fp_para_cfg))
                        d_p_list.append((fp_d_p, fp_d_p_cfg))
                        state = self.state[(fp_para, fp_para_cfg)]

                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])
                else:
                    raise Exception("para grad is None!")

            F.sgd(params_with_grad,
                  self.bit_width_list,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
