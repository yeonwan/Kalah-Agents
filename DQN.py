import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1)  # 6 x 9 => 5 x 8
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1) # 5 x 8 => 4 x 7
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2) # 4 x 7 => 2 x 5
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1) # 4 x 7 => 2 x 5
        self.bn4 = nn.BatchNorm2d(512)
        linear_input_size = 1024
        self.fc1 = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    # convolution input
    # 보드 2
    # 내 보드 1
    # 적 보드 1
    # 0 pad
    # 내 스코어 3
    # 0 pad
    # 적 스코어 3