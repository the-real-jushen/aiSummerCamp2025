'''
Homework:
Methodology for solving image classification problems.
Train a simple convolutional neural network (CNN) to classify CIFAR images.
'''

# %%
# importing
import torch
import numpy as np
from matplotlib import pyplot as plt
# %%
# load CIFAR-10 data
from torchvision import datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
train_images, train_labels = train_dataset.data, train_dataset.targets
test_images, test_labels = test_dataset.data, test_dataset.targets
# %%
# Code here!
# inspect images

# %%
# prepocess dataset (include dataloader)
'''
1. one-hot encoded lavels:Transforming the label value into a 10 element binary vector   
(sometimes dont't needed, eg: when using "sparse_categorical_crossentropy" as the loss function in model.compile, we don't need to transfrom the labels, cause this loss function expects integer labels and it does the one-hot encoding it self(sparse_))  
2. Normalize the pixal data, scaling the pixel data from [0,255] to range [0,1]
'''

# %%
# create a CNN model

# %%
# train the model

# %%
# evaluate the model