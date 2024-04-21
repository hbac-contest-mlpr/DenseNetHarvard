from model import DenseNet
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch
import torch.nn as nn
import dataloader
import time

model = DenseNet(layer_num=(6,12,24,16),growth_rate=32,in_channels=1,classes=6)

loss_fn = nn.KLDivLoss(reduction="batchmean") # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer
model= model.to(device)
PATH = "./saved_models/model_15.pth"
checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

# write data code 


with torch.inference_mode():
    test_pred = model(x_test)
    test_loss = loss_fn(test_pred,y_test)
    print(f"KL div loss: {test_loss:.5f}")