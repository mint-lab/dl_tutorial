import torch

A = torch.tensor([[1., 1.], [4., 1.]])
b = torch.tensor([[4.], [2.]])
A_inv = A.inverse() # Note) np.linalg.inv(A)
print(A_inv.mm(b))  # Note) np.matmul(A_inv, b)
