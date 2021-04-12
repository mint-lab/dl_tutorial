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
USE_CUDA = torch.cuda.is_available() # Try False for 'cpu'
EPOCHS = 10000
EPOCH_PRINT = 1000
BATCH_SIZE = 50

# Load a dataset partially
iris = datasets.load_iris()
iris.data = iris.data[:,0:2]
iris.feature_names = iris.feature_names[0:2]
iris.color = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
input_size, output_size = len(iris.feature_names), len(iris.target_names)

# Load the dataset as torch.utils.data.TensorDataset
x = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)
data = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# Define a model
class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function
def train(model, train_loader, loss_fn, optimizer, dev='cpu'):
    model.train() # Notify layers (e.g. DropOut, BatchNorm) that it's the training mode
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)

# Define the evaluation function
def evaluate(model, test_loader, loss_fn, dev='cpu'):
    model.eval() # Notify layers (e.g. DropOut, BatchNorm) that it's the evaluation mode
    test_loss, n_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            output = model(x)
            loss = loss_fn(output, y)
            y_pred = torch.argmax(output, dim=1)

            test_loss += loss.item()
            n_correct += (y == y_pred).sum().item()
    return test_loss / len(test_loader.dataset), n_correct / len(test_loader.dataset)

# Train the model
dev = torch.device('cuda' if USE_CUDA else 'cpu')
model = TwoLayerNN().to(dev)
loss_fn = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_list = []
start = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, loss_fn, optimizer, dev)
    test_loss, test_accuracy = evaluate(model, train_loader, loss_fn, dev)

    loss_list.append([epoch, train_loss, test_loss, test_accuracy])
    if epoch % EPOCH_PRINT == 0:
        print(f'{epoch:>6} ({time.time()-start:.0f} sec), TrainLoss={train_loss:.6f}, TestLoss={test_loss:.6f}, TestAccuracy={test_accuracy:.3f}')
elapse = time.time() - start

# Visualize the training and test losses
plt.title(f'Training and Test Losses (time: {elapse:.3f} [sec] @ CUDA: {USE_CUDA})')
loss_array = np.array(loss_list)
plt.plot(loss_array[:,0], loss_array[:,1], label='Training Loss')
plt.plot(loss_array[:,0], loss_array[:,2], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss values')
plt.xlim(loss_array[0,0], loss_array[-1,0])
plt.grid()
plt.legend()
plt.show()

# Visualize training results (decision boundaries)
x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:, 0].max() + 1
y_min, y_max = iris.data[:, 1].min() - 1, iris.data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack((xx.flatten(), yy.flatten())).T
xy_tensor = torch.from_numpy(xy).float().to(dev)
zz = torch.argmax(model(xy_tensor), dim=1).cpu().detach().numpy()
plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=ListedColormap(iris.color), alpha=0.2)

# Visualize testing results
plt.title(f'Fully-connected NN (accuracy: {loss_array[-1,-1]:.3f})')
predict = torch.argmax(model(x.to(dev)), dim=1).cpu().detach().numpy()
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target], edgecolors=iris.color[predict])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
