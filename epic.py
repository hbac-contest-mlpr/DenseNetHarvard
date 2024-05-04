import torch
import torch.nn as nn


X = torch.randn(10, 100, 1001)   
print(X.shape)
Z=nn.AvgPool1d(3)(X)
Y=nn.MaxPool1d(3)(X)
print(Y.shape)
print(X.shape[-1]//3)
X = nn.Conv1d(100, 100, 3, padding=1)(Z+Y)
print(X.shape)