from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import dataloader
import time
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


n_epochs = 100
lr = 0.001

torch.manual_seed(0)
X,Y = dataloader.load_data(10,4)


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=6) # model

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
model= model.to(device)

x_train = torch.from_numpy(x_train).unsqueeze(1).to(device) # 320,1000 to 320,1,1000
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).unsqueeze(1).to(device)
y_test = torch.from_numpy(y_test).to(device)
y_train = nn.functional.log_softmax(y_train,dim=1)
y_test = nn.functional.log_softmax(y_test,dim=1)




#to do 
for epoch in range(n_epochs):
    t0 = time.time()
    y_pred = model(x_train) #forward pass 
    print(y_pred[0])
    print(y_train[0])
    loss = loss_fn(y_pred,y_train) # calculate loss
    print(loss)
    break 
    loss.backward() # backward pass
    
    optimizer.step() # update weights
     
    model.eval() # set model to evaluation mode
    
    with torch.inference_mode(): 
        test_pred = model(x_test) # forward pass
        
        test_loss = loss_fn(test_pred,y_test) # calculate loss
        
        
    # if epoch % 100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f} | Time: {time.time()-t0:.2f} s")