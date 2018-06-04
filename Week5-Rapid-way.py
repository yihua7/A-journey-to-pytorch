import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Method 1
class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_input,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.output(x)
        return x
net1=Net(1,10,1)

# Method 2
net2=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
        )

print(net1)
print(net2)

# Prepare for fake data
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

loss_func=torch.nn.MSELoss()
optim=torch.optim.SGD(net2.parameters(),lr=0.1)

for i in range(100):
    prediction=net2.forward(x)
    loss=loss_func(prediction,y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i%5==0:
        print("%dth " %i,"train, loss=%f" %loss.data[0])

