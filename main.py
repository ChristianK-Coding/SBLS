import math
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plot

# import torch_directml
import torch
import torch.nn as nn

import datetime

from bad_sbls import BadSBLS
from sbls2 import SBLS2

# os.system('clear')

# torch.set_printoptions(profile="full")

# dml = torch_directml.device()

def name(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]

def show_img(img):
    img = np.array(img, dtype='float')
    # pixels = img.reshape((28, 28))
    plot.imshow(img, cmap='gray', aspect="auto", interpolation='none')
    plot.show()

# import dataset
num_steps = 30
batch_size_train = 10000
batch_size_test = 1000
data_path = '/home/christian/data/mnist'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# create dataloader
train_loader = DataLoader(mnist_train, batch_size=batch_size_train, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=True, drop_last=True)

# instantiate loss function
loss = nn.CrossEntropyLoss()


# slicing the train data up into data and targets manually
loss_history = []
t1 = datetime.datetime.now()


sbls = SBLS2(input_size=28*28, output_size=10, simulation_steps=num_steps, initital_feature_size=100, initial_enhancement_size=2000)

# train_input, train_target = next(iter(train_loader))

# for i in range(100):
#     print(f"loop {i}")
#     train_output = sbls(train_input)

#     _, indices = torch.max(train_output, 1)
#     print(f"Accuracy: {(indices == train_target).sum() / batch_size_train}%")
#     print(f"Loss: {loss(train_output, train_target)}")

#     sbls.add_new_data((train_input, train_target))


for count, data in enumerate(train_loader):
    print(f"loop {count}")

    # Testing accuracy
    test_input, test_targets = next(iter(test_loader))
    # print(test_targets[0:10])

    test_output = sbls(test_input)
    # print(test_output[0:10])

    values, indices = torch.max(test_output, 1)
    print(f"accuracy:{(indices == test_targets).sum() / batch_size_test}")
    # print(indices[0:10])

    test_loss = loss(test_output, test_targets)
    print(f"loss in this loop: {test_loss}")
    loss_history.append(test_loss)

    sbls.add_new_data(data)






t2 = datetime.datetime.now()

print(f"Took {t2 - t1}")

print(loss_history)