from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
import dataloader
import time
from torch.utils.data import DataLoader, TensorDataset
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

torch.cuda.empty_cache()
n_epochs = 1000
lr = 0.001
batch_size = 32

torch.manual_seed(0)
X,Y = dataloader.load_data(100,3)
X = torch.from_numpy(X).unsqueeze(1).to(device)
Y = torch.from_numpy(Y).to(device)
dataset = TensorDataset(X,Y)
train_dataset , test_dataset = torch.utils.data.random_split(dataset, [0.8,0.2])
train_batches = DataLoader(train_dataset, batch_size=batch_size)
test_batches = DataLoader(test_dataset, batch_size=batch_size)

model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=6) # model

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
model= model.to(device)

 
for epoch in range(n_epochs):
    print(f"Epoch: {epoch}")
    t0 = time.time()
    
    model.train()
    
    for batch_idx, (x_train, y_train) in enumerate(train_batches):     
        y_pred = model(x_train) #forward pass 
        loss = loss_fn(y_pred,y_train) # calculate loss
        optimizer.zero_grad() # zero the gradients

        loss.backward() # backward pass

        optimizer.step() # update weights

        if batch_idx % 25 == 0:
            print(f"\t Batch_idx: {batch_idx} | Loss: {loss:.5f}")
    model.eval()    # set model to evaluation mode
    with torch.inference_mode(): 
        test_loss = 0
        for batch_idx, (x_test, y_test) in enumerate(test_batches):
            test_pred = model(x_test) # forward pass
            test_loss += loss_fn(test_pred,y_test) # calculate loss
        test_loss /= len(test_batches)
    print(f"Time: {time.time()-t0:.2f} s | Test loss: {test_loss:.5f} ")
    
    #saving checkpoint
    if epoch and epoch % 15 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            }, f"./saved_models/model_{epoch}.pth")