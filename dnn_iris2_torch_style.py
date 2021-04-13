#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
import time

# Define hyperparameters
EPOCH_MAX = 10000
EPOCH_LOG = 1000
BATCH_SIZE = 50
OPT_LEARN_RATE = 0.01
USE_CUDA = torch.cuda.is_available() # Try False for 'cpu'

# A Two-layer NN model
HIDDEN_SIZE = 4
class MyDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train a model with the given batches
def train(model, batch_data, loss_fn, optimizer):
    model.train()
    train_loss, n_data = 0, 0
    dev = next(model.parameters()).device
    for batch_idx, (x, y) in enumerate(batch_data):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_data += len(y)
    return train_loss / n_data

# Evaluate a model
def evaluate(model, batch_data, loss_fn):
    model.eval()
    test_loss, n_correct, n_data = 0, 0, 0
    with torch.no_grad():
        dev = next(model.parameters()).device
        for x, y in batch_data:
            x, y = x.to(dev), y.to(dev)
            output = model(x)
            loss = loss_fn(output, y)
            y_pred = torch.argmax(output, dim=1)

            test_loss += loss.item()
            n_correct += (y == y_pred).sum().item()
            n_data += len(y)
    return test_loss / n_data, n_correct / n_data



if __name__ == '__main__':
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1.1. Load the Iris dataset partially
    iris = datasets.load_iris()
    iris.data = iris.data[:,0:2]
    iris.feature_names = iris.feature_names[0:2]
    iris.color = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # 1.2. Wrap the dataset with torch.utils.data.DataLoader
    x = torch.tensor(iris.data, dtype=torch.float32, device=dev)
    y = torch.tensor(iris.target, dtype=torch.long, device=dev)
    train_data = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Instantiate a model, loss function, and optimizer
    model = MyDNN(len(iris.feature_names), len(iris.target_names)).to(dev)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=OPT_LEARN_RATE)

    # 3. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, train_loader, loss_fn, optimizer)
        valid_loss, valid_accuracy = evaluate(model, train_loader, loss_fn)

        loss_list.append([epoch, train_loss, valid_loss, valid_accuracy])
        if epoch % EPOCH_LOG == 0:
            print(f'{epoch:>6} ({(time.time()-start)/60:.2f} min), TrainLoss={train_loss:.6f}, ValidLoss={valid_loss:.6f}, ValidAcc={valid_accuracy:.3f}')
    elapse = time.time() - start

    # 4.1. Visualize the loss curves
    plt.title(f'Training and Validation Losses (time: {elapse/60:.2f} [min] @ CUDA: {USE_CUDA})')
    loss_array = np.array(loss_list)
    plt.plot(loss_array[:,0], loss_array[:,1], label='Training Loss')
    plt.plot(loss_array[:,0], loss_array[:,2], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.xlim(loss_array[0,0], loss_array[-1,0])
    plt.grid()
    plt.legend()
    plt.show()

    # 4.2. Visualize training results (decision boundaries)
    x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:, 0].max() + 1
    y_min, y_max = iris.data[:, 1].min() - 1, iris.data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    xy = np.vstack((xx.flatten(), yy.flatten())).T
    xy_tensor = torch.from_numpy(xy).float().to(dev)
    zz = torch.argmax(model(xy_tensor), dim=1).cpu().detach().numpy()
    plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=ListedColormap(iris.color), alpha=0.2)

    # 4.3. Visualize testing results
    plt.title(f'Fully-connected NN (accuracy: {loss_array[-1,-1]:.3f})')
    predict = torch.argmax(model(x.to(dev)), dim=1).cpu().detach().numpy()
    plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target], edgecolors=iris.color[predict])
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()
