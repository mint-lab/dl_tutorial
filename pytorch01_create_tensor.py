import numpy as np
import torch

# 1. Create a tensor from a composite data
x = np.array([[3, 29, 82], [10, 18, 84]])
y = torch.tensor(x)
print(y.ndim, y.dim())      # 2                       cf. x.ndim
print(y.nelement())         # 6                       cf. x.size
print(y.shape, y.size())    # torch.Size([2, 3])      cf. x.shape
print(y.dtype)              # torch.int32             cf. x.dtype

# 2. Create a tensor using initializers
p = torch.rand(3, 2)        # Try zeros, ones, eyes, empty, arange, linspace,
q = torch.zeros_like(p)     #     and their ..._like
print(p.dtype)              # torch.float32
print(q.shape)              # torch.Size([3, 2])

# 3. Interpret as a tensor (generating only a view)
z = torch.as_tensor(x)      # Or torch.from_numpy(x)  cf. np.asarray()
x[-1,-1] = 86
print(z[-1])                # tensor([10, 18, 86])

# 4. Access elements
print(y[:,1])               # tensor([29, 18])
print(y[0,0])               # tensor(3)               cf. x[0,0] == 3
print(y[0,0].item())        # 3

# 5. CUDA tensors
if torch.cuda.is_available():
    print(y.device)         # 'cpu'
    y_cuda = y.cuda()       # Or y.to('cuda')
    print(y_cuda.device)    # 'cuda:0'
    y_cpu = y_cuda.cpu()    # Or y.cuda.to('cpu')
    print(y_cpu.device)     # 'cpu'

    x_cpu = y_cpu.numpy()   # Or np.array(y_cpu)
    x_cuda = y_cuda.numpy() # Error!
