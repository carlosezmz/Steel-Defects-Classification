# DeepNets_Project
Updates (5/6/20)



Work Done:

-  tried the basic CNN model for the classification task, with approx 7000 defects (4 categories) sample from 'train.csv' file

-  trained the model for different epochs (6,10) and batch size (100, 64, 128)

Observation:

- Training accuracy achieved about 92% with 10 epochs, testing accuracy of 88%

- there is overfitting to this model due to small training examples



Results of current work:

6 epochs, 100 batch size , 256 x 256 x 3 -> 64, 128, 256, flatten (dense)accuracy : .8550, .8789, .8832, .8857, .8905, .8955

10 epochs, 64 batch size , 256 x 256 x 3 -> 64, 128, 256, flatten (dense)accuracy: 0.8593, .8810, .8861, .8874, .8887, .8971, .8993, .9041, .9137, .9192testing accuracy: 88.33

10 epochs, 128 batch size, 256 x 256 x 3 -> 64, 128, 256, flatten (dense)accuracy: .8475, .8779, .8841, .8885, .8931, .8972, .9023, .9082, .9167, .9189testing accuracy: 88.1





Work to be Done:

- the no-defect data is to be added from folder 'train_images'.  ( approx 7000 + 5000)

- Data Augmentation is to be applied, and a better algorithm is to be tried over this (ResNet)
