from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split
device  = 'cuda' if torch.cuda.is_available() else 'cpu'


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

n_epochs = 1
lr = 0.001

torch.manual_seed(0)
m1 = MultivariateNormal(torch.zeros(1), torch.eye(1))
m2 = MultivariateNormal(torch.ones(1)*5, torch.eye(1))
m3 = MultivariateNormal(torch.ones(1)*10, torch.eye(1))
class1 = m1.sample(sample_shape=torch.Size([100]))
class2 = m2.sample(sample_shape=torch.Size([100]))
class3 = m3.sample(sample_shape=torch.Size([100]))
X = torch.cat([class1,class2,class3],dim=0).to(device)
Y = torch.cat([torch.zeros(100),torch.ones(100),torch.ones(100)*2]).to(device) # make this probability distribution of 3 classes
x_train,y_train,x_test,y_test = train_test_split(X,Y,test_size=0.2)


model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=3) # model

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=lr) # optimizer
model= model.to(device)
x_train= torch.reshape(x_train,(1,1,-1))
y_train= torch.reshape(y_train,(1,-1))
x_test= torch.reshape(x_test,(1,1,-1))
y_test= torch.reshape(y_test,(1,-1))
for epoch in range(n_epochs):
    
    y_pred = model(x_train) #forward pass 
    print(y_pred)
    # loss = loss_fn(y_pred,y_train) # calculate loss
    # acc = accuracy_fn(y_true=y_train,  y_pred=y_pred) # calculate accuracy
    
    # loss.backward() # backward pass
    
    # optimizer.step() # update weights
     
    # model.eval() # set model to evaluation mode
    
    # with torch.inference_mode(): 
    #     test_pred = model(x_test) # forward pass
        
    #     test_loss = loss_fn(test_pred,y_test) # calculate loss
    #     test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred) # calculate accuracy
        
        
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")