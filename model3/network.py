import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        dropout_rate = 0.01
        super(Net, self).__init__()

        # Input image size : 28x28

        # Convolution Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate)
        ) # Output Size - 26, Receptive Field - 3
        
        # Convolution Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        ) # Output Size - 24, Receptive Field - 5
        
        # Transition Layer 1 - MaxPool and 1x1 conv
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=1, padding=0, bias=False),
        ) # Output Size - 12, Receptive Field - 6

        # Convolution Layer 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        ) # Output Size - 10, Receptive Field -10 

        # Convolution Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 12, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate)
        ) # Output Size - 8, Receptive Field - 14

        
        # Convolution Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        ) # Output Size - 6, Receptive Field - 18

        # Convolution Layer 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate)
        ) # Output Size - 4, Receptive Field - 22
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10) 
        return F.log_softmax(x, dim=-1)