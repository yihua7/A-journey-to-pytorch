import torch

# Creat variables
x=torch.autograd.Variable(torch.Tensor([1]),requires_grad=True)
w=torch.autograd.Variable(torch.Tensor([2]),requires_grad=True)
b=torch.autograd.Variable(torch.Tensor([3]),requires_grad=True)

y=w*x+b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)
