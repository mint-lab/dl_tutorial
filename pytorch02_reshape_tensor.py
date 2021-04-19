import numpy as np
import torch

x = np.array([[[29, 3], [18, 10]], [[27, 10], [12, 5]]])
y = torch.tensor(x)
print(y.ndim)           # 3
print(y.shape)          # torch.Size([2, 2, 2])

print(y[0])             # tensor([[29, 3], [18, 10]])

p = y.view(-1)          # tensor([29, 3, 18, 10, 27, 10, 12, 5])
print(p.shape, p)       # torch.Size([8])
q = y.view(1, -1)       # tensor([[29, 3, 18, 10, 27, 10, 12, 5]])
print(q.shape, q)       # torch.Size([1, 8])
r = y.view(2, -1)       # tensor([[29, 3, 18, 10], [27, 10, 12, 5]])
print(r.shape, r)       # torch.Size([2, 4])
# cf. tensor also supports 'reshape' which is same with 'view'.
s = y.reshape(2, -1, 1) # tensor([[[29], [3], [18], [10]], [[27], [10], [12], [5]]])
print(s.shape, s)       # torch.Size([2, 4, 1])

ss = s.squeeze(2)       # cf. s.squeeze(0) and s.squeeze(1) are effectless.
print(ss)               # tensor([[29, 3, 18, 10], [27, 10, 12, 5]])
u0 = ss.unsqueeze(0)    # tensor([[[29, 3, 18, 10], [27, 10, 12,  5]]])
print(u0.shape, u0)     # torch.Size([1, 2, 4])
u1 = ss.unsqueeze(1)    # tensor([[[29, 3, 18, 10]], [[27, 10, 12, 5]]])
print(u1.shape, u1)     # torch.Size([2, 1, 4])
u2 = ss.unsqueeze(2)    # tensor([[[29], [3], [18], [10]], [[27], [10], [12], [5]]])
print(u2.shape, u2)     # torch.Size([2, 4, 1])

# Switch indices each other, (i, j) to (j, i)
t_021 = y.transpose(1, 2)           # tensor([[[29, 18],
print(t_021, t_021.is_contiguous()) #          [ 3, 10]], ... ]) False
c_021 = t_021.contiguous()          # tensor([[[29, 18],
print(c_021, c_021.is_contiguous()) #          [ 3, 10]], ... ]) True
t_102 = y.transpose(0, 1)           # tensor([[[29,  3],
print(t_102, t_102.is_contiguous()) #           27, 10]], ... ]) False

# Assign indices
t_201 = y.permute(2, 0, 1)          # tensor([[[29, 18],
print(t_201, t_201.is_contiguous()) #          [27, 12]], ... ]) False

# cf. Reshaping does not copy contents.
y[0,0,0] = 27
print(p)                            # tensor([27, 3, 18, 10, 27, 10, 12, 5])
print(t_021)                        # tensor([[[27, 18], ...], ...])
print(c_021)                        # tensor([[[29, 18], ...], ...])

# Copy a tensor and detach it from its connected computational graph
z = y.clone().detach()              # cf. x.clone()
y[0,0,0] = 1
print(y)                            # tensor([[[ 1, 3], ...], ...])
print(z)                            # tensor([[[27, 3], ...], ...])
