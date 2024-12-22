import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        dropout_rate = 0.07
        super(Net, self).__init__()
        # Input image size : 28x28

        # Convolution Layer 1 - Output Size - 26
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=0, bias=False), # Input Channels - 1, Output Channels - 12
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        # Convolution Layer 2 - Output Size - 24
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        # Convolution Layer 3 - Output Size - 22
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 12, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Transition Layer 1 - MaxPool and 1x1 conv - Output Size - 11
        self.transition1 = nn.Sequential(
            nn.Conv2d(12, 10, kernel_size=1, padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # Convolution Layer 4 - Output Size - 9
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 12, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Transition Layer 2 - MaxPool and 1x1 conv - Output Size - 4
        #self.transition2 = nn.Sequential(
        #    nn.MaxPool2d(2, 2),
        #    nn.Conv2d(16, 10, kernel_size=1, padding=0, base=False),
        #)
        
        # Convolution Layer 5 - Output Size - 7
        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Convolution Layer 6 - Output Size - 5
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 10, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transition1(x)

        x = self.conv4(x)
        #x = self.transition2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10) 
        return F.log_softmax(x, dim=-1)
