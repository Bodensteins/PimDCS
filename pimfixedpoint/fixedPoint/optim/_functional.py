from typing import List, Tuple, Optional
import torch
from ..nn import fixedPointArithmetic as fpA


def sgd(params: List[Tuple],
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

        fpA.fixed_point_add_(param, d_p, alpha=-lr)
