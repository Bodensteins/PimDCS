from mapping.archFunctional import ArchLevel
from mapping.archMapping import DefaultMapStrategy
from mapping.archRecord import Arch
from mnist_model import PimConvMnist
from performance.performance import PerformanceManager
from mapping.archConst import archConst


def test_performance():
    net = PimConvMnist()
    """
    shapes of each layer in this model.
    |-- input                                                                               [28, 28]            
    |-- Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))                                    [10, 24, 24]        
    |-- ReLU()                                                                              [20, 8, 8]          
    |-- MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)     [20, 4, 4]          
    |-- Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))                                   [20, 8, 8]          
    |-- ReLU()                                                                              [20, 8, 8]          
    |-- MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)     [20, 4, 4]          
    |-- Flatten(start_dim=1, end_dim=-1)                                                    [1, 320]            
    |-- Quan()                                                                              [1, 320]            
    |-- Dropout()                                                                           [1, 320]            
    |-- Linear(in_features=320, out_features=10, bias=True)                                 [1, 10]             
    |-- DeQuan()                                                                            [1, 10]   
    """
    print(net)
    ac = Arch(4, 16, 8, 4, 1)
    mapStrategy = DefaultMapStrategy(ac)

    performanceManager = PerformanceManager(net, 8, archConst.PIMRunMode.inference, mapStrategy)
    performanceManager.areaModule.printArch()
    performanceManager.areaModule.print()
    for layer_name, cfg in performanceManager.layer_record.items():
        print(layer_name)
        print(cfg.array_names)
        for array_name in cfg.array_names:
            print(mapStrategy.getLogicalArrayMap(array_name))


def test_mapping():
    ac = Arch(4, 16, 8, 1, 128)
    pid = int("11000100001", 2)
    print(pid)
    aid = ac.pid2Archid(pid)
    print(aid)
    print(ac.Archid2pid(aid))

    print(ac.is_same_level(64, 65, ArchLevel.tile_level))

    dmap = DefaultMapStrategy(ac)
    dmap.current_pid = 99
    dmap.allocLogicalArray(5, (9, 9), 8)
    temp = dmap.getLogicalArrayMap(5)
    print(temp)
    dmap.printByLevel(99, archLevel=ArchLevel.pe_level)
    print(dmap.archRecord.get_free_pe_pid_list(64, ArchLevel.bank_level))


if __name__ == "__main__":
    # test_performance()
    test_mapping()
