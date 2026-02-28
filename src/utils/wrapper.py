import torch
import torch.nn as nn
from src.nn_modules.fp_conv import FpConv2d
from src.nn_modules.fp_linear import FpLinear
from config.globalCfg import globalCfg

def gen_fp_conv(conv:nn.Conv2d):
    fp_conv = FpConv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, 
                        kernel_size=conv.kernel_size, stride=conv.stride, 
                        padding=conv.padding, dilation=conv.dilation, 
                        groups=conv.groups, padding_mode=conv.padding_mode)
    fp_conv.weight = conv.weight
    if hasattr(conv,'bias'):
        fp_conv.bias = conv.bias
        fp_conv.has_bias = True
    return fp_conv


def gen_fp_linear(linear:nn.Linear):
    fp_linear = FpLinear(in_features=linear.in_features, out_features=linear.out_features)
    fp_linear.weight = linear.weight
    if hasattr(linear,'bias'):
        fp_linear.bias = linear.bias
        fp_linear.has_bias = True
    return fp_linear


def fold_bn_into_conv(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        # beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * (conv_module.bias - y_mean) / safe_std + bn_module.bias
        else:
            bias = - bn_module.weight * y_mean / safe_std + bn_module.bias
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    w, b = weight, bias
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b.data)
        conv_module.has_bias = True
    else:
        conv_module.bias.data = b.data
    conv_module.weight.data = w.data


def wrap_modules(model):
    if globalCfg.mmManagerTpye != 0 \
            and globalCfg.mmManagerTpye != 4 \
            and globalCfg.mmManagerTpye != 5 \
            and globalCfg.mmManagerTpye != 6:
        return

    wrapped_modules_dict = {}
    prev_layer_name = None
    module_dict = {}
    
    for name, m in [_ for _ in model.named_modules()]:
        module_dict[name] = m
        new_m = None
        if isinstance(m, nn.Conv2d):
            new_m = gen_fp_conv(m)
        elif isinstance(m, nn.Linear):
            new_m = gen_fp_linear(m)
        elif isinstance(m, nn.BatchNorm2d):
            if prev_layer_name in wrapped_modules_dict:
                prev_layer = wrapped_modules_dict[prev_layer_name]
                if isinstance(prev_layer, FpConv2d):
                    fold_bn_into_conv(prev_layer, m)
                    new_m = nn.Identity()
                    # new_m = m
        if new_m is not None:
            idx=name.rfind('.')
            if idx==-1:
                setattr(model,name,new_m)
                print(f"set {name} as {type(new_m)}")
            else:
                father_name = name[:idx]
                father_module = module_dict[father_name]
                setattr(father_module, name[idx+1:], new_m)
                print(f"set {name} in {father_name} as {type(new_m)}")
            wrapped_modules_dict[name] = new_m
        prev_layer_name = name
        
    layer_no = 0
    for name, module in wrapped_modules_dict.items():
        module.name = name
        if isinstance(module, FpConv2d) or isinstance(module, FpLinear):
            module.create_mat_mul_manager()
            module.set_layer_no(layer_no)
            layer_no += 1

