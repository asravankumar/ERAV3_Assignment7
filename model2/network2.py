import torch
import torch.nn as nn
import torch.nn.functional as F


# second model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input image size : 28x28

        # Convolution Layer 1 - Output Size - 26
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=0, bias=False), # Input Channels - 1, Output Channels - 12
            nn.BatchNorm2d(10),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Convolution Layer 2 - Output Size - 24
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        # Transition Layer 1 - MaxPool and 1x1 conv - Output Size - 12
        self.transition1 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # Convolution Layer 3 - Output Size - 10
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Convolution Layer 4 - Output Size - 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Transition Layer 2 - MaxPool and 1x1 conv - Output Size - 4
        #self.transition2 = nn.Sequential(
        #    nn.MaxPool2d(2, 2),
        #    nn.Conv2d(16, 10, kernel_size=1, padding=0, base=False),
        #)
        
        # Convolution Layer 5 - Output Size - 6
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        # Convolution Layer 6 - Output Size - 4
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.transition2(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.gap(x)
        x = x.view(-1, 10) 
        return F.log_softmax(x, dim=-1)




'''
(era3_assignments) sravan@sravan-latitude-3410 ~/Personal/Courses/ERA3/Assignment7/ERAV3_Assignment7/model2 (main)$ python train.py 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
           Dropout-3           [-1, 10, 26, 26]               0
              ReLU-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,440
       BatchNorm2d-6           [-1, 16, 24, 24]              32
           Dropout-7           [-1, 16, 24, 24]               0
              ReLU-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             160
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
      BatchNorm2d-12           [-1, 16, 10, 10]              32
          Dropout-13           [-1, 16, 10, 10]               0
             ReLU-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
      BatchNorm2d-16             [-1, 16, 8, 8]              32
          Dropout-17             [-1, 16, 8, 8]               0
             ReLU-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
      BatchNorm2d-20             [-1, 16, 6, 6]              32
          Dropout-21             [-1, 16, 6, 6]               0
             ReLU-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 10, 4, 4]           1,440
      BatchNorm2d-24             [-1, 10, 4, 4]              20
          Dropout-25             [-1, 10, 4, 4]               0
             ReLU-26             [-1, 10, 4, 4]               0
AdaptiveAvgPool2d-27             [-1, 10, 1, 1]               0
================================================================
Total params: 9,346
Trainable params: 9,346
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.65
Params size (MB): 0.04
Estimated Total Size (MB): 0.68
----------------------------------------------------------------
--------------------------------------------
EPOCH: 0
Loss=0.20734256505966187 Batch_id=468 Accuracy=91.42: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.94it/s]

Training set: Average loss: 0.2073, Accuracy: 54852/60000 (91.42%)


Test set: Average loss: 0.1464, Accuracy: 9656/10000 (96.56%)

--------------------------------------------
--------------------------------------------
EPOCH: 1
Loss=0.12842045724391937 Batch_id=468 Accuracy=97.31: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:35<00:00, 13.04it/s]

Training set: Average loss: 0.1284, Accuracy: 58387/60000 (97.31%)


Test set: Average loss: 0.0949, Accuracy: 9748/10000 (97.48%)

--------------------------------------------
--------------------------------------------
EPOCH: 2
Loss=0.05767567455768585 Batch_id=468 Accuracy=97.88: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.85it/s]

Training set: Average loss: 0.0577, Accuracy: 58726/60000 (97.88%)


Test set: Average loss: 0.0605, Accuracy: 9847/10000 (98.47%)

--------------------------------------------
--------------------------------------------
EPOCH: 3
Loss=0.06936610490083694 Batch_id=468 Accuracy=98.18: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.77it/s]

Training set: Average loss: 0.0694, Accuracy: 58907/60000 (98.18%)


Test set: Average loss: 0.0521, Accuracy: 9859/10000 (98.59%)

--------------------------------------------
--------------------------------------------
EPOCH: 4
Loss=0.07197003811597824 Batch_id=468 Accuracy=98.22: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.88it/s]

Training set: Average loss: 0.0720, Accuracy: 58932/60000 (98.22%)


Test set: Average loss: 0.0524, Accuracy: 9843/10000 (98.43%)

--------------------------------------------
--------------------------------------------
EPOCH: 5
Loss=0.04481805860996246 Batch_id=468 Accuracy=98.48: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:38<00:00, 12.14it/s]

Training set: Average loss: 0.0448, Accuracy: 59088/60000 (98.48%)


Test set: Average loss: 0.0459, Accuracy: 9859/10000 (98.59%)

--------------------------------------------
--------------------------------------------
EPOCH: 6
Loss=0.04076337814331055 Batch_id=468 Accuracy=98.49: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:43<00:00, 10.78it/s]

Training set: Average loss: 0.0408, Accuracy: 59092/60000 (98.49%)


Test set: Average loss: 0.0372, Accuracy: 9897/10000 (98.97%)

--------------------------------------------
--------------------------------------------
EPOCH: 7
Loss=0.014137146063148975 Batch_id=468 Accuracy=98.66: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.72it/s]

Training set: Average loss: 0.0141, Accuracy: 59193/60000 (98.66%)


Test set: Average loss: 0.0334, Accuracy: 9906/10000 (99.06%)

--------------------------------------------
--------------------------------------------
EPOCH: 8
Loss=0.024642430245876312 Batch_id=468 Accuracy=98.70: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:37<00:00, 12.49it/s]

Training set: Average loss: 0.0246, Accuracy: 59219/60000 (98.70%)


Test set: Average loss: 0.0382, Accuracy: 9878/10000 (98.78%)

--------------------------------------------
--------------------------------------------
EPOCH: 9
Loss=0.10300072282552719 Batch_id=468 Accuracy=98.72: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.94it/s]

Training set: Average loss: 0.1030, Accuracy: 59229/60000 (98.72%)


Test set: Average loss: 0.0309, Accuracy: 9915/10000 (99.15%)

--------------------------------------------
--------------------------------------------
EPOCH: 10
Loss=0.015084912069141865 Batch_id=468 Accuracy=98.78: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.81it/s]

Training set: Average loss: 0.0151, Accuracy: 59267/60000 (98.78%)


Test set: Average loss: 0.0261, Accuracy: 9927/10000 (99.27%)

--------------------------------------------
--------------------------------------------
EPOCH: 11
Loss=0.10448773950338364 Batch_id=468 Accuracy=98.74: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:37<00:00, 12.54it/s]

Training set: Average loss: 0.1045, Accuracy: 59246/60000 (98.74%)


Test set: Average loss: 0.0246, Accuracy: 9929/10000 (99.29%)

--------------------------------------------
--------------------------------------------
EPOCH: 12
Loss=0.02876495011150837 Batch_id=468 Accuracy=98.78: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.87it/s]

Training set: Average loss: 0.0288, Accuracy: 59268/60000 (98.78%)


Test set: Average loss: 0.0276, Accuracy: 9913/10000 (99.13%)

--------------------------------------------
--------------------------------------------
EPOCH: 13
Loss=0.05093611776828766 Batch_id=468 Accuracy=98.87: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.84it/s]

Training set: Average loss: 0.0509, Accuracy: 59322/60000 (98.87%)


Test set: Average loss: 0.0280, Accuracy: 9911/10000 (99.11%)

--------------------------------------------
--------------------------------------------
EPOCH: 14
Loss=0.026586269959807396 Batch_id=468 Accuracy=98.85: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:37<00:00, 12.40it/s]

Training set: Average loss: 0.0266, Accuracy: 59312/60000 (98.85%)


Test set: Average loss: 0.0255, Accuracy: 9926/10000 (99.26%)

--------------------------------------------
--------------------------------------------
EPOCH: 15
Loss=0.008895652368664742 Batch_id=468 Accuracy=98.86: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:39<00:00, 11.96it/s]

Training set: Average loss: 0.0089, Accuracy: 59314/60000 (98.86%)


Test set: Average loss: 0.0239, Accuracy: 9930/10000 (99.30%)

--------------------------------------------
--------------------------------------------
EPOCH: 16
Loss=0.021864451467990875 Batch_id=468 Accuracy=98.97: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:41<00:00, 11.20it/s]

Training set: Average loss: 0.0219, Accuracy: 59385/60000 (98.97%)


Test set: Average loss: 0.0260, Accuracy: 9926/10000 (99.26%)

--------------------------------------------
--------------------------------------------
EPOCH: 17
Loss=0.03270135447382927 Batch_id=468 Accuracy=98.99: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:39<00:00, 11.92it/s]

Training set: Average loss: 0.0327, Accuracy: 59396/60000 (98.99%)


Test set: Average loss: 0.0254, Accuracy: 9921/10000 (99.21%)

--------------------------------------------
--------------------------------------------
EPOCH: 18
Loss=0.028773993253707886 Batch_id=468 Accuracy=98.94: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:36<00:00, 12.76it/s]

Training set: Average loss: 0.0288, Accuracy: 59365/60000 (98.94%)


Test set: Average loss: 0.0331, Accuracy: 9901/10000 (99.01%)

--------------------------------------------
--------------------------------------------
EPOCH: 19
Loss=0.023462045937776566 Batch_id=292 Accuracy=98.95:  62%|████████████████████████████████████████████████████████████████████                                         | 293/469 [00:27<00:13, 13.28it/s]Loss=0.08933306485414505 Batch_id=468 Accuracy=98.93: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:40<00:00, 11.50it/s]

Training set: Average loss: 0.0893, Accuracy: 59357/60000 (98.93%)


Test set: Average loss: 0.0271, Accuracy: 9912/10000 (99.12%)

--------------------------------------------
(era3_assignments) sravan@sravan-latitude-3410 ~/Personal/Courses/ERA3/Assignment7/ERAV3_Assignment7/model2 (main)$
'''