import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
x1=torch.normal(-2*n_data,1)
y0=torch.zeros(100)
y1=torch.ones(100)

x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.FloatTensor)

x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,cmap='RdYlGn')

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)
    def forward(x):
        x=F.relu(self.hidden(x))
        x=self.output(x)
        return x

net=Net(n_feature=2,n_hidden=10,n_output=2)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.02)

loss_func=torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2==0:
        # Show learning process
        plt.cla()
        prediction=torch.max(out,1)[1]
        pred_y=prediction.data.numpy()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y)/200
        plt.text(1.5,-4,'Accuracy=%.2f' % accuracy, fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
