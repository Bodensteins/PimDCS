from src.nn_modules.fp_conv import FpConv2d
from src.nn_modules.fp_linear import FpLinear
from config.globalCfg import globalCfg

def save_statistic(model, save_path:str=None, postfix:str='test'):
   if globalCfg.mmManagerTpye != 4 and globalCfg.mmManagerTpye != 5 and globalCfg.mmManagerTpye != 6:
        return
   print(save_path)
   for name, m in [_ for _ in model.named_modules()]:
        if isinstance(m, FpConv2d) or isinstance(m, FpLinear):
            print(f"save statistic of {name}, layer_no: {m.layer_no}")
            m.save_statistic(save_path, postfix)

