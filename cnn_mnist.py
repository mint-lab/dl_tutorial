#!/usr/bin/python
#-*- coding: utf-8 -*-
"""Digit Classification with CNN and the MNIST Dataset

This example is modified and improve from the Pytorch official example as follows:
- Link: https://github.com/pytorch/examples
"""

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import time, PIL
from sklearn import metrics
from dnn_iris2 import train, evaluate

# Define hyperparameters
EPOCH_MAX = 50
EPOCH_LOG = 1
OPTIMIZER_PARAM = { 'lr': 0.01 }
SCHEDULER_PARAM = { 'step_size': 10, 'gamma': 0.5 }
DATA_PATH = './data'
DATA_LOADER_PARAM = { 'batch_size': 100, 'shuffle': True }
USE_CUDA = torch.cuda.is_available()
SAVE_MODEL = 'cnn_mnist.pt' # Make empty('') if you don't want save the model
RANDOM_SEED = 777

# A four-layer CNN model
# - Try more or less layers, channels, and kernel size
# - Try to apply batch normalization (e.g. 'nn.BatchNorm' and 'nn.BatchNorm2d')
# - Try to apply skip connection (used in ResNet)
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Notation:    (batch_size, channel, height, width)
        # Input :      (batch_size,  1, 28, 28)
        # Layer1: conv (batch_size, 32, 28, 28)
        #         pool (batch_size, 32, 14, 14)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Layer2: conv (batch_size, 64, 14, 14)
        #         pool (batch_size, 64,  7,  7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.2)

        # Input :      (batch_size, 64*7*7)
        # Layer3: fc   (batch_size, 512)
        self.fc3   = nn.Linear(64*7*7, 512)
        self.drop3 = nn.Dropout(0.2)

        # Layer4: fc   (batch_size, 10)
        self.fc4   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc3(x))
        x = self.drop3(x)

        x = F.log_softmax(self.fc4(x), dim=1)
        return x

# Predict a digit using the given model
def predict(image, model):
    model.eval()
    with torch.no_grad():
        # Convert the given image to its 1 x 1 x 28 x 28 tensor
        if type(image) is torch.Tensor:
            tensor = image.type(torch.float) / 255  # Normalize to [0, 1]
        else:
            tensor = 1 - TF.to_tensor(image)        # Invert (white to black)
        if tensor.ndim < 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 3:
            tensor = TF.rgb_to_grayscale(tensor)    # Make grayscale
        tensor = TF.resize(tensor, 28)              # Resize to 28 x 28
        dev = next(model.parameters()).device
        tensor = tensor.unsqueeze(0).to(dev)        # Add onw more dims

        output = model(tensor)
        digit = torch.argmax(output, dim=1)
        return digit.item()

if __name__ == '__main__':
    # 0. Preparation
    torch.manual_seed(RANDOM_SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOM_SEED)
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1. Load the MNIST dataset
    preproc = torchvision.transforms.ToTensor()
    data_train = torchvision.datasets.MNIST(DATA_PATH, train=True,  download=True, transform=preproc)
    data_valid = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=preproc)
    loader_train = torch.utils.data.DataLoader(data_train, **DATA_LOADER_PARAM)
    loader_valid = torch.utils.data.DataLoader(data_valid, **DATA_LOADER_PARAM)

    # 2. Instantiate a model, loss function, and optimizer
    model = MyCNN().to(dev)
    loss_func = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), **OPTIMIZER_PARAM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **SCHEDULER_PARAM)

    # 3.1. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, loader_train, loss_func, optimizer)
        valid_loss, valid_accuracy = evaluate(model, loader_valid, loss_func)
        scheduler.step()

        loss_list.append([epoch, train_loss, valid_loss, valid_accuracy])
        if epoch % EPOCH_LOG == 0:
            elapse = (time.time() - start) / 60
            print(f'{epoch:>6} ({elapse:>6.2f} min), TrLoss={train_loss:.6f}, VaLoss={valid_loss:.6f}, VaAcc={valid_accuracy:.3f}, lr={scheduler.get_last_lr()}')
    elapse = (time.time() - start) / 60

    # 3.2. Save the trained model if necessary
    if SAVE_MODEL:
        torch.save(model.state_dict(), SAVE_MODEL)

    # 4.1. Visualize the loss curves
    plt.title(f'Training and Validation Losses (time: {elapse:.2f} [min] @ CUDA: {USE_CUDA})')
    loss_array = np.array(loss_list)
    plt.plot(loss_array[:,0], loss_array[:,1], label='Training Loss')
    plt.plot(loss_array[:,0], loss_array[:,2], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.xlim(loss_array[0,0], loss_array[-1,0])
    plt.grid()
    plt.legend()
    plt.show()

    # 4.2. Visualize the confusion matrix
    predicts = [predict(datum, model) for datum in data_valid.data]
    conf_mat = metrics.confusion_matrix(data_valid.targets, predicts)
    conf_fig = metrics.ConfusionMatrixDisplay(conf_mat)
    conf_fig.plot()

    # 5. Test your image
    print(predict(data_train.data[0], model)) # 5
    with PIL.Image.open('data/cnn_mnist_test.png').convert('L') as image:
        print(predict(image, model))          # 3
