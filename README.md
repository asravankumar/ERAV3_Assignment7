# ERAV3_Assignment7

## Model - 1

### Target:
  1. Get the Set-up Right
  2. Set Transforms, Data Loader, Basic Working Code, Basic Training & Test Loop.
  3. Set Basic Skeleton Right
  4. Make the model lighter, use MaxPool at required places.

### Results:
  1. Parameters: 10,984
  2. Best Train Accuracy: 99.35
  3. Best Test Accuracy: 98.95

### Analysis:
  1. A lighter model with fully connected layer at the end. Scope of improvement while replacing it with GAP and also reduces the number of paramters.
  2. Model is definitely overfitting after epoch 6 when the training accuracy started showing higher accuracy than testing accuracy. and the gap widenened in later epochs.
  3. Need to use regularization to overcome overfitting and use batch normalization to converge faster and improve model efficiency.

Please go to [Model-1](https://github.com/asravankumar/ERAV3_Assignment7/tree/main/model1) for a detailed information, receptive field calculations and training logs.

## Model - 2
### Target:
  1. Solve the problem of overfitting by using regularization techniques and under model parameters of 8000.

### Results:
  1. Parameters: 7,878
  2. Best Train Accuracy: 99.00%
  3. Best Test Accuracy: 99.23%

### Analysis:
  1. The model is not over fitting.
  2. Use of Regularization techniques like droput to overcome overfitting and was achieved. Dropout of 0.1 was initially tried but 0.7 yielded better training and test accuracies.
  3. Earlier, maxpool was used in two layers but reducing it to one increased both the training and test accuracies by 1% and could easily reach ~99.2% by doing so.
  4. Perhaps, we should add image augmentation techniques to improve the model and also a better scheduler to achieve the best plateau during training.

Please go to [Model-2](https://github.com/asravankumar/ERAV3_Assignment7/tree/main/model2) for a detailed information, receptive field calculations and training logs.

## Model - 3 (The Final One)
### Target:
  1. Under 8000 parameters and under 15 epochs, consistently reach test accuracy of 99.4% at least three times.

### Results:
  1. Parameters: 7,830
  2. Best Train Accuracy within epoch 15: 98.75% (epoch 15)
  3. Best Test Accuracy within epoch 15: 99.45% (epoch 13)
  4. Reached 99.45% at epoch 12 and at epoch 13, 14, 15 were 99.43%, 99.44% and 99.45% respectively.

### Analysis:
  1. Upon adding image augmentation techniques like rotation and affine rotation, the model was further underfitting now and also improved the test accuracy.
  2. Modified the number of channels in each layer multiple times, played with different learning rates but finally found that at almost every network with number of parameters ranging from 7500 to 8000, at epoch 7, 8, 9, 10 it reaches test accuracy of 99.25-99.40 but after that it does not improve significantly.
  3. Finally, with a simple StepLR scheduler and after 9th epoch, the lr got updated and it reached the desired test accuracies(>= 99.40%) consistently.
  4. Then reducing the dropout to 0.01 has improved the training and test accuracy to a large extent. Perhaps, a larger droput in a small dataset like MNIST is loosing out crucial information.
  5. Training the same model in GPU yielded slightly better results after epoch 10.

Please go to [Model-3](https://github.com/asravankumar/ERAV3_Assignment7/tree/main/model3) for a detailed information, receptive field calculations and training logs.