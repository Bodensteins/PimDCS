# -*- coding: utf-8 -*-
from . import _functional as fpF
import torch.optim._functional as F
from ..nn import fixedPointArithmetic as fpA
import torch
from torch.optim.optimizer import Optimizer
from .optimizer import OptimMode


class SGD(Optimizer):
    def __init__(self, named_params, bit_width_list: list, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False,
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

                fpF.sgd(params_with_grad,
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
    #             raise ValueError("parameter group didn't specify a value of required optimization parameter " +
    #                              name)
    #         else:
    #             param_group.setdefault(name, default)
    #
    #     params = param_group['params']
    #     if len(params) != len(set(params)):
    #         warnings.warn("optimizer contains a parameter group with duplicate parameters; "
    #                       "in future, this will cause an error; "
    #                       "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)
    #
    #     param_set = set()
    #     for group in self.param_groups:
    #         param_set.update(set(group['params']))
    #
    #     if not param_set.isdisjoint(set(param_group['params'])):
    #         raise ValueError("some parameters appear in more than one parameter group")
    #
    #     self.param_groups.append(param_group)
