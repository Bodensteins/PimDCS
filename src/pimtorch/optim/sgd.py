# -*- coding: utf-8 -*-
import torch
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer
from typing import List, Tuple, Optional
from src.pimtorch.config.globalCfg import OptimMode
from src.pimtorch.nn import fixedPointArithmetic as fpA


def sgd(params: List[Tuple],
        params_bit_width_list: List[int],
        d_p_list: List[Tuple],
        momentum_buffer_list: List[Optional[Tuple]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            fpA.fixed_point_add_(d_p, param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                fp_d_p, fp_d_p_cfg = d_p
                buf = (torch.clone(fp_d_p).detach(), torch.clone(fp_d_p_cfg).detach())
                momentum_buffer_list[i] = buf
            else:
                # here has bugs, we don't update the value of momentum_buffer_list
                fpA.fixed_point_mul_(buf, momentum)
                # buf = fpA.mul_num(list(buf), momentum)
                fpA.fixed_point_add_(buf, d_p, alpha=1-dampening)

            if nesterov:
                fpA.fixed_point_add_(d_p, buf, alpha=momentum)
            else:
                d_p = buf

        fpA.fixed_point_add_(param, d_p, source_bit_width=params_bit_width_list[i], alpha=-lr)

class SGD(Optimizer):
    def __init__(self, named_params, bit_width_list: list, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 run_mode=OptimMode.FullFix):
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

        params = []
        fp_params = []
        torch_params = []
        for name, param in named_params:
            if 'fp_weight' in name:
                fp_params.append(param)
            else:
                torch_params.append(param)

        # for param_tuple in zip(fp_weight, fp_weight_cfg):
        #     fp_params.append(param_tuple)
        if len(fp_params) % 2 != 0:
            raise Exception("Illegal fp params numbers")

        params.append({"params": fp_params, "is_fixed_point_params": True})
        params.append({"params": torch_params, "is_fixed_point_params": False})

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
            if group["is_fixed_point_params"]:
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

                sgd(params_with_grad,
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
            else:
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

                try:
                    F.sgd(params_with_grad,
                        d_p_list,
                        momentum_buffer_list,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        lr=lr,
                        dampening=dampening,
                        nesterov=nesterov)
                except TypeError as e:
                    if "maximize" in str(e):
                        F.sgd(params_with_grad,
                            d_p_list,
                            momentum_buffer_list,
                            weight_decay=weight_decay,
                            momentum=momentum,
                            lr=lr,
                            dampening=dampening,
                            nesterov=nesterov,
                            maximize = False)
                    else:
                        raise TypeError(e)

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
        return loss

    def zero_grad(self, set_to_none: bool = True):
        super(SGD, self).zero_grad(set_to_none=set_to_none)

    # todo: later to support it
    # def add_fp_param_group(self, param_group):
    #     r"""Add a param group to the :class:`Optimizer` s `param_groups`.
    #
    #     This can be useful when fine tuning a pre-trained network as frozen layers can be made
    #     trainable and added to the :class:`Optimizer` as training progresses.
    #
    #     Args:
    #         param_group (dict): Specifies what Tensors should be optimized along with group
    #         specific optimization options.
    #     """
    #     assert isinstance(param_group, dict), "param group must be a dict"
    #
    #     params = param_group['params']
    #     if isinstance(params, torch.Tensor):
    #         param_group['params'] = [params]
    #     elif isinstance(params, set):
    #         raise TypeError('optimizer parameters need to be organized in ordered collections, but '
    #                         'the ordering of tensors in sets will change between runs. Please use a list instead.')
    #     else:
    #         param_group['params'] = list(params)
    #
    #     for param in param_group['params']:
    #         if not isinstance(param, torch.Tensor):
    #             raise TypeError("optimizer can only optimize Tensors, "
    #                             "but one of the params is " + torch.typename(param))
    #         if not param.is_leaf:
    #             raise ValueError("can't optimize a non-leaf Tensor")
    #
    #     for name, default in self.defaults.items():
    #         if default is required and name not in param_group:
    #             raise ValueError("config group didn't specify a value of required optimization config " +
    #                              name)
    #         else:
    #             param_group.setdefault(name, default)
    #
    #     params = param_group['params']
    #     if len(params) != len(set(params)):
    #         warnings.warn("optimizer contains a config group with duplicate parameters; "
    #                       "in future, this will cause an error; "
    #                       "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)
    #
    #     param_set = set()
    #     for group in self.param_groups:
    #         param_set.update(set(group['params']))
    #
    #     if not param_set.isdisjoint(set(param_group['params'])):
    #         raise ValueError("some parameters appear in more than one config group")
    #
    #     self.param_groups.append(param_group)
