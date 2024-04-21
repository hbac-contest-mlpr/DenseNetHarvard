from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
import dataloader
import time
from torchsummary import summary
import torchinfo

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=6)
torchinfo.summary(model)
lr = 0.001

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
model= model.to(device)
PATH = "./saved_models/model_45.pth"
checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

# write data code 
X,Y = dataloader.load_data(1,3)
X = torch.from_numpy(X).unsqueeze(1).to(device)
Y = torch.from_numpy(Y).to(device)
with torch.inference_mode():
    test_pred = model(X)
    test_loss = loss_fn(test_pred,Y)
    print(f"KL div loss: {test_loss:.5f}")