import numpy as np
import matplotlib.pyplot as plt
import torch

x_trian=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
                  [9.779],[6.182],[7.59],[2.167],[7.042],
                  [10.791],[5.313],[7.997],[3.1]],dtype=np.float32)
y_trian=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],
                  [3.366],[2.596],[2.53],[1.221],[2.827],
                  [3.465],[1.65],[2.904],[1.3]],dtype=np.float32)

def make_features(x):
    """ Build features i.e. a matrix with columns [x,x^2,x^3]."""
    x=x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

W_target=torch.FloatTensor([0.5,3,2.4]).unsqueeze(1)
b_target=torch.FloatTensor([0.9])

def f(x):
    """Approximated function."""
    return x.mm(W_target)+b_target[0]

def get_batch(batch_size=32):
    """Builds a batch i.e. (x,f(x)) pair."""
    random=torch.randn(batch_size)
    x=make_features(random)
    y=f(x)
    if torch.cuda.is_available():
        return torch.autograd.Variable(x).cuda(),torch.autograd.Variable(y).cuda()
    else:
        return torch.autograd.Variable(x),torch.autograd.Variable(y)

# Define module
class poly_model(torch.nn.Module):
    def __init__(self):
        super(poly_model,self).__init__()
        self.poly=torch.nn.Linear(3,1)

    def forward(self,x):
        out=self.poly(x)
        return out

if torch.cuda.is_available():
    model=poly_model().cuda()
else:
    model=poly_model()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

epoch=0
while True:
    # Get data
    batch_x,batch_y=get_batch()
    # Forward pass
    output=model(batch_x)
    loss=criterion(output,batch_y)
    print_loss=loss.data[0]
    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update parameter
    optimizer.step()
    epoch+=1
    if epoch%100==0:
        print("loss="+str(print_loss))
    if print_loss<1e-3:
        break

weight=model.parameters()
print(weight)
print(W_target,b_target)

# Test
# model.eval()
# x,y=get_batch()
# x_axi=x[0,0]
# prediction=model(x)
# prediction=prediction.data.numpy()
# plt.plot(np.array(x_axi),np.array(y),'ro',label='Original data')
# plt.plot(np.array(x_axi),np.array(prediction),label='Fitting Line')
# plt.show()


