import torch
import torch.nn as nn
import matplotlib.pyplot as plt

activation_funcs = [
    {'name': 'Sigmoid',    'func': nn.Sigmoid()},
    {'name': 'Tanh',       'func': nn.Tanh()},
    {'name': 'ReLU',       'func': nn.ReLU()},
    {'name': 'Leaky ReLU', 'func': nn.LeakyReLU(0.1)},
    {'name': 'ELU',        'func': nn.ELU()},
    # Try more activation functions
]

for act in activation_funcs:
    x = torch.linspace(-10, 10, 200, requires_grad=True)
    y = act['func'](x)
    y.sum().backward()

    plt.title(act['name'])
    x_np, y_np, grad = x.detach().numpy(), y.detach().numpy(), x.grad.numpy()
    plt.plot(x_np, y_np, label='$\phi(x)$')
    plt.plot(x_np, grad, label='$\partial \phi(x) / \partial x$')
    plt.grid()
    plt.legend()
    plt.show()
