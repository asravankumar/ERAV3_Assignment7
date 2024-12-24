import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input image size : 28x28

        # Convolution Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1, bias=False), # Input Channels - 1, Output Channels - 12
            nn.ReLU()
        ) # Output Size - 28, Receptive Field - 3
        
        # Convolution Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # Output Size - 28, Receptive Field - 5
        
        # Transition Layer 1 - MaxPool and 1x1 conv
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False),
        ) # Output Size - 14, Receptive Field - 6

        # Convolution Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # Output Size - 14, Receptive Field - 10

        # Convolution Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # Output Size - 14, Receptive Field - 14

        # Transition Layer 2 - MaxPool and 1x1 conv
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False),
        ) # Output Size - 7, Receptive Field - 16
        
        # Convolution Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # Output Size - 7, Receptive Field - 24

        # Convolution Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # Output Size - 5, Receptive Field - 32
        
        self.fc = nn.Linear(10 * 5 * 5, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.transition1(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.transition2(x)

        x = self.conv5(x)
        x = self.conv6(x)
  
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
