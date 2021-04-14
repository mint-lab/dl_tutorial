import torch

A = torch.tensor([[1., 1.], [4., 1.]])
b = torch.tensor([[4.], [2.]])
A_inv = A.inverse() # cf. np.linalg.inv(A)
print(A_inv.mm(b))  # cf. np.matmul(A_inv, b)
