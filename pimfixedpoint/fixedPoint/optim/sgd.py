# -*- coding: utf-8 -*-
import types
import torch
from ..nn import fixedPointArithmetic as fpA
from torch.optim.optimizer import Optimizer
import torch.optim._functional as F
from .optimizer import OptimMode


class SGD(Optimizer):
    def __init__(self, named_parameters, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 run_mode=OptimMode.full_fix):
        if not isinstance(named_parameters, types.GeneratorType) or named_parameters.__name__ != "named_parameters":
            raise TypeError("PimSGD expected generator that return by named_parameters(), but got %s" % type(
                named_parameters).__name__)
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.runMode = run_mode

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        params = []

        fp_weight_list = []
        fp_weight_cfg_list = []
        torch_param_groups = []

        for name, param in named_parameters:
            if 'fp_weight_cfg' in name:
                fp_weight_cfg_list.append(param)
            elif 'fp_weight' in name:
                fp_weight_list.append(param)
            else:
                print('here is named_parameters')
                print(name, param.shape)
                torch_param_groups.append(param)

        for layer_params in zip(fp_weight_list, fp_weight_cfg_list):
            params.append({"params": list(layer_params), "is_fixed_point_params": True})
        params.append({"params": torch_param_groups, "is_fixed_point_params": False})

        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["is_fixed_point_params"]:
                # PIM SGD, each group is single pim layer
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                lr = group['lr']

                fp_weight, fp_weight_cfg = group["params"]
                d_w = fp_weight.grad
                d_w_cfg = fp_weight_cfg.grad
                if d_w is not None and d_w_cfg is not None:
                    if self.runMode == OptimMode.full_fix:
                        sub_weight = fpA.mul_num([d_w, d_w_cfg], -lr)
                        fpA.add_alpha_tensor_([fp_weight, fp_weight_cfg], sub_weight)
                        fp_weight.grad = None  # todo: this should in zero method, not here
                    elif self.runMode == OptimMode.float_weight:
                        print("unsupported optimizer mode")
                        pass
                    elif self.runMode == OptimMode.full_float:
                        print("unsupported optimizer mode")
                        pass
                else:
                    raise Exception("err: grad is None")
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
