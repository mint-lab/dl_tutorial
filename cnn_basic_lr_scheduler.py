import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

base_lr = 1.
epoch_max = 100
schedulers = [
    {'name': 'StepLR',       'class': lr_scheduler.StepLR,      'param': {'step_size': 10, 'gamma': 0.5}},
    {'name': 'LambdaLR',     'class': lr_scheduler.LambdaLR,    'param': {'lr_lambda': lambda epoch: 0.5**(epoch / 10)}},
    {'name': 'MultiStepLR',  'class': lr_scheduler.MultiStepLR, 'param': {'milestones': [40, 60, 70, 75, 80, 85, 90, 95], 'gamma': 0.5}},
    {'name': 'OneCycleLR',   'class': lr_scheduler.OneCycleLR,  'param': {'max_lr': base_lr, 'total_steps': epoch_max}},
    {'name': 'CosAnnealing', 'class': lr_scheduler.CosineAnnealingWarmRestarts, 'param': {'T_0': 50}},
    # Try more learning rate schedulers
]

for sch in schedulers:
    x = torch.tensor(1., requires_grad=True)            # A dummy parameter
    optimizer = torch.optim.SGD([x], lr=base_lr)        # Instantiate an optimizer
    scheduler = sch['class'](optimizer, **sch['param']) # Instantiate a LR scheduler
    lr_values = []
    for i in range(epoch_max):
        optimizer.step()
        scheduler.step()
        lr_values.append(optimizer.param_groups[0]['lr'])
    plt.plot(range(epoch_max), lr_values, label=sch['name'])

plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.show()