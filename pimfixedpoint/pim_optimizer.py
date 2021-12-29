# -*- coding: utf-8 -*-
import types

import torch
import quantization
from quantization import add_alpha_tensor_, mul_num, de_quantization, remove_additional_col, \
    add_additional_col_of_zero, write_array_
from enum import Enum
from torch.optim.optimizer import Optimizer
import torch.optim._functional as F

class OptimMode(Enum):
    full_fix = 0
    float_weight = 1
    full_float = 2


class PimSGD(Optimizer):
    def __init__(self, named_parameters, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, run_mode=OptimMode.full_fix):
        if not isinstance(named_parameters, types.GeneratorType) or named_parameters.__name__ != "named_parameters":
            raise TypeError("PimSGD expected generator that return by named_parameters(), but got %s" % type(named_parameters).__name__)
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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        params = []

        wtArray = []
        wArray = []
        wArray_cfgs = []
        wtArray_cfgs = []
        delta_qweight_t_cfgs = []
        weight = []
        torch_param_groups = []

        for name, param in named_parameters:
            if 'pim_wArr_cfg' in name:
                wArray_cfgs.append(param)
            elif 'pim_wtArr_cfg' in name:
                wtArray_cfgs.append(param)
            elif 'pim_delta_qweight_t_cfg' in name:
                delta_qweight_t_cfgs.append(param)
            elif 'pim_wArr' in name:
                wArray.append(param)
            elif 'pim_wtArr' in name:
                wtArray.append(param)
            elif 'pim_weight' in name:
                weight.append(param)
            else:
                torch_param_groups.append(param)

        for layer_params in zip(wArray, wArray_cfgs, wtArray, wtArray_cfgs, weight, delta_qweight_t_cfgs):
            params.append({"params": list(layer_params), "is_pim_params": True})
        params.append({"params": torch_param_groups, "is_pim_params": False})

        super(PimSGD, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["is_pim_params"]:
                # PIM SGD, each group is single pim layer

                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                lr = group['lr']

                wArr, wArr_cfg, wtArr, wtArr_cfg, weight, d_wt_cfg = group["params"]
                d_wt = wtArr.grad
                if d_wt is not None:
                    if self.runMode == OptimMode.full_fix:
                        delta_wt = mul_num([d_wt, d_wt_cfg], -lr)
                        add_alpha_tensor_([wtArr, wtArr_cfg], delta_wt)
                        d_w = add_additional_col_of_zero([remove_additional_col(d_wt).t(), d_wt_cfg])
                        delta_w = mul_num([d_w, d_wt_cfg], -lr)
                        add_alpha_tensor_([wArr, wArr_cfg], delta_w)
                        wtArr.grad = None
                    elif self.runMode == OptimMode.float_weight:
                        weight_grad = de_quantization(mul_num([remove_additional_col(d_wt).t(), d_wt_cfg], -lr))
                        weight.add_(weight_grad)
                        temp = quantization.creat_quantization_para(bit_width=16,
                                                                    tensor_type=quantization.TensorType.Normal,
                                                                    device=d_wt.device)
                        x = quantization.quantization_tensor(temp, weight)
                        write_array_([wArr, wArr_cfg], [x, temp])
                        x = quantization.quantization_tensor(temp, weight.t())
                        write_array_([wtArr, wtArr_cfg], [x, temp])
                        wtArr.grad = None
                        weight.grad = None
                    elif self.runMode == OptimMode.full_float:
                        weight.add_(weight.grad * (-lr))
                        temp = quantization.creat_quantization_para(bit_width=16,
                                                                    tensor_type=quantization.TensorType.Normal)
                        x = quantization.quantization_tensor(temp, weight)
                        write_array_([wArr, wArr_cfg], [x, temp])
                        x = quantization.quantization_tensor(temp, weight.t())
                        write_array_([wtArr, wtArr_cfg], [x, temp])
                        wtArr.grad = None
                        weight.grad = None
            else:
                # Pytorch SGD

                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                lr = group['lr']

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)

                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])

                F.sgd(params_with_grad,
                      d_p_list,
                      momentum_buffer_list,
                      weight_decay=weight_decay,
                      momentum=momentum,
                      lr=lr,
                      dampening=dampening,
                      nesterov=nesterov)

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer

        return loss
