import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 conv layers 2 maxpool layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Cnn():
    return Net()

if __name__=="__main__":
    net = Net()
    print(torchsummary.summary(net, input_size=(3, 32, 32)))