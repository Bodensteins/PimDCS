import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),  # CIFAR-10 的图像为 32x32, 因此使用 3x3 卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),  # 根据 CIFAR-10 图像大小调整全连接层的输入维度
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 10),  # 最后一层的输出节点数为类别数（10）
        )
        
    def forward(self, x):
        x = self.alexnet(x)
        return x
    
    def eval(self):
        res = super().eval()
        return res


