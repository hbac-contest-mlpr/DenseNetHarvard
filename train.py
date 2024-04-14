from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import dataloader
device  = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

n_epochs = 1
lr = 0.001

torch.manual_seed(0)
X,Y = dataloader.load_data(100,4)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print(y_train.shape)
model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=6) # model

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
model= model.to(device)
x_train= torch.from_numpy(x_train).unsqueeze(1) # 320,1000 to 320,1,1000
# y_train= torch.from_numpy(y_train).unsqueeze(1)
x_test= torch.from_numpy(x_test).unsqueeze(1)
# y_test= torch.from_numpy(y_test).unsqueeze(1)
#to do 
# for epoch in range(n_epochs):
    
#     y_pred = model(x_train) #forward pass 
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