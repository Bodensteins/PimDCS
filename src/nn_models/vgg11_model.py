import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vgg11 = nn.Sequential(nn.Conv2d(3, 64, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                                   nn.Flatten(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(512, num_classes)
                                   )
      

    def forward(self, x):
        x = self.vgg11(x)
        return x

    def eval(self):
        res = super().eval()
        return res


