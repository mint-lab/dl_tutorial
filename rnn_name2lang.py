#!/usr/bin/python
#-*- coding: utf-8 -*-
"""Name-to-Language Classification with Character-level RNN

This example are modified and improved from the PyTorch official tutorial as follows:
- Link: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import unicodedata, os
import glob, time, random
from dnn_iris2_torch_style import train

# Define global constants
__LETTER_DICT__ = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"

# Define hyperparameters
EPOCH_MAX = 10
EPOCH_LOG = 1
OPT_LEARN_RATE = 0.005
USE_CUDA = torch.cuda.is_available() # Try False for 'cpu'
DATA_PATH = './data/names/*.txt'

# Convert Unicode to ASCII
# e.g. Ślusàrski to Slusarski
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in __LETTER_DICT__)

# Read raw files which contain names belong to each language
# cf. Each filename is used as its target's name.
def load_name_dataset(files):
    data = []
    target = []
    target_names = []
    for idx, filename in enumerate(files):
        lang = os.path.splitext(os.path.basename(filename))[0]
        names = open(filename, encoding='utf-8').read().strip().split('\n')

        data += [unicode2ascii(name) for name in names]
        target += [idx] * len(names)
        target_names.append(lang)
    return data, target, target_names

# Transform the given text to its one-hot encoded tensor
# cf. Tensor size: len(text) x 1 x len(__LETTER_DICT__)
#                  sequence_length x batch_size x input_size
def text2onehot(text, device='cpu'):
    tensor = torch.zeros(len(text), 1, len(__LETTER_DICT__), device=device)
    for idx, letter in enumerate(text):
        tensor[idx][0][__LETTER_DICT__.find(letter)] = 1
    return tensor

# A simple RNN model
# - Try a different RNN unit such LSTM and GRU
# - Try less or more hidden units
# - Try more layers (e.g. 'num_layers=2') and dropout (e.g. 'dropout=0.4')
HIDDEN_SIZE = 128
class MyRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size, HIDDEN_SIZE)
        self.fc  = torch.nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        output, hidden = self.rnn(x)
        x = self.fc(output[-1]) # Use output of the last sequence
        return x

# Predict a result of the given datum
def predict(text, model, target_names, n_predict=5):
    print(f'* Name: {text}')
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        text_tensor = text2onehot(text, dev)
        output = model(text_tensor)[0]
        prob = nn.functional.softmax(output, dim=0)
        top_val, top_idx = prob.topk(n_predict) # Get top N predictions
        for i in range(len(top_val)):
            print(f'  {i+1}. {target_names[top_idx[i].item()]:<10}: {top_val[i]*100:4.1f} %')



if __name__ == '__main__':
    dev = torch.device('cuda' if USE_CUDA else 'cpu')

    # 1. Load the name2lang dataset
    data, target, target_names = load_name_dataset(glob.glob(DATA_PATH))
    data_all = [(text2onehot(data[i], device=dev), torch.LongTensor([target[i]]).to(dev)) for i in range(len(data))]
    random.shuffle(data_all)

    # 2. Instantiate a model, loss function, and optimizer
    model = MyRNN(len(__LETTER_DICT__), len(target_names)).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=OPT_LEARN_RATE)

    # 3. Train the model
    loss_list = []
    start = time.time()
    for epoch in range(1, EPOCH_MAX + 1):
        train_loss = train(model, data_all, loss_fn, optimizer)

        loss_list.append([epoch, train_loss])
        if epoch % EPOCH_LOG == 0:
            print(f'{epoch:>6} ({(time.time()-start)/60:.2f} min), TrainLoss={train_loss:.6f}')
    elapse = time.time() - start

    # 4. Visualize the loss curves
    plt.title(f'Training Loss (time: {elapse:.3f} [sec] @ CUDA: {USE_CUDA})')
    loss_array = np.array(loss_list)
    plt.plot(loss_array[:,0], loss_array[:,1], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.xlim(loss_array[0,0], loss_array[-1,0])
    plt.grid()
    plt.legend()
    plt.show()

    # 5. Test your texts
    predict('Choi', model, target_names)
    predict('Jane', model, target_names)
    predict('Daniel', model, target_names)
    predict('Chow', model, target_names)
    predict('Tanaka', model, target_names)
