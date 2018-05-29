import torch

from torch.autograd import Variable  # Type Variable's gradient can be calculated.
import torch.nn.functional as F      # Activation functions are all here: torch.nn.functional
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)  # Transform the dimesion of x to 1*100
y=x.pow(2)+0.2*torch.rand(x.size())

x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):   # Initialization
        super(Net,self).__init__()                    # Inherit
        self.hidden=torch.nn.Linear(n_feature,n_hidden)  # torch.nn.Linear : output=A*input+B
        self.predict=torch.nn.Linear(n_hidden,n_output)  # Another linear layer

    def forward(self,x):
        x=F.relu(self.hidden(x))  # Use relu func from torch.nn.functional
        x=self.predict(x)
        return x

net=Net(1,10,1)
print(net)

plt.ion()
plt.show()

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)  # Input net's parameters to optimizer
loss_func=torch.nn.MSELoss()

for t in range(100):
    prediction=net(x)
    loss=loss_func(prediction,y)  # prediction is ahead of y
    
    optimizer.zero_grad()   # Initialize
    loss.backward()         # Calculating gradient of net
    optimizer.step()        # Subtract lr*grad
    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f' % loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

