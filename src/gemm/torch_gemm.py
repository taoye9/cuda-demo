import torch

A = torch.arange(16).view(4, 4)
print(A)

B = torch.arange(16).view(4, 4)

print(B)


C= A @ B
print(C)
