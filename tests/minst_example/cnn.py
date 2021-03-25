import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, c1=3, k1=6, k2=2):
        super(Net, self).__init__()
        print("c1=", c1)
        self.conv1 = nn.Conv2d(c1, k1, 5)
        self.pool = nn.MaxPool2d(k2, k2)
        self.conv2 = nn.Conv2d(k1, 5, 5)
        self.conv3 = nn.Conv2d(5, 10, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(2).squeeze(2)
        return x
