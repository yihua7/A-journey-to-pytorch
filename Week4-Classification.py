import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),0).type(torch.LongTensor) # Labels in torch must be LongTensor type

x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_input,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.output(x)
        return x

net=Net(2,10,2)

optimizer=torch.optim.SGD(net.parameters(),lr=0.005)
loss_func=torch.nn.CrossEntropyLoss()

for i in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%2==0:
        plt.cla()
        # torch.max(input,dim)
        # dim==0 -> returns the maximum value of each column
        # dim==1 -> returns the maximum value of each row
        # [0] fetch the maximum value
        # [1] fetch the position of maximum value
        prediction=torch.max(F.softmax(out),1)[1]
        pred_y=prediction.data.numpy()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y)/200.
        plt.text(1.5,-4,'Accuracy=%.2f' %accuracy, fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

