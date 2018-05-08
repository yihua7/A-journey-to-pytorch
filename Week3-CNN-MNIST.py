import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

if not(os.path.exists('./mnist/')) or not os.list.dir('./mnist/'):
    DOWNLOAD_MNIST=True

train_data=torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD,
        )

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i' %train_data.train_labels[0])
plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x=Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
                nn.Conv2(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                )
        self.conv2=nn.Sequential(
                nn.Conv2d(16,32,5,1,2),
                nn.ReLU(),
                nn.MaxPool3d(2),
                )
        self.out=nn.Linear(32*7*7,10)

        def forward(self,x):
            x=self.conv1(x)
            x=self.conv2(x)
            x=x.view(x.size(0),-1)
            output=self.out(x)
            return output,x

cnn=CNN()
print(cnn)

optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK=True
except: HAS_SK=False; print('Plwase install for layer visualization')

def plot_with_labels(lowDweights,labels):
    plt.cla()
    X,Y=lowDWeights[:,0],lowDWeights[:,1]
    for x,y,s in zip(X,Y,labels):
        c=cm.rainbow(int(255*s/9)); plt.text(x,y,s,backgroundcolor=c,fontsize=9)
    plt.xlim(X.min(),X.max());

