# https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch/blob/main/DenseNet.py
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self,in_channels,middle_channels=128,out_channels=32): # should we put middle_channels here?
        
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels,middle_channels,1),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(middle_channels,out_channels,3,padding=1)
        )
    
    def forward(self,x):
        return torch.cat([x,self.layer(x)],dim=1)

class DenseBlock(nn.Sequential):
    def __init__(self,layer_num,growth_rate,in_channels,middle_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels+i*growth_rate,middle_channels,growth_rate)
            self.add_module('denselayer%d'%(i),layer)

class Transition(nn.Sequential):
    def __init__(self,channels):
        super(Transition, self).__init__()
        self.add_module('norm',nn.BatchNorm1d(channels))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv1d(channels,channels//2,3,padding=1))
        self.add_module('Avgpool',nn.AvgPool1d(2))

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                                nn.BatchNorm1d(out_channels),
                                nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class DenseNet(nn.Module):
    def __init__(self,layer_num=(6,12,24,16),growth_rate=32,init_features=64,in_channels=1,middle_channels=128,classes=5):
        super(DenseNet, self).__init__()
        self.feature_channel_num=init_features
        self.conv=nn.Conv1d(in_channels,self.feature_channel_num,7,2,3)
        self.norm=nn.BatchNorm1d(self.feature_channel_num)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool1d(3,2,1)

        self.DenseBlock1=DenseBlock(layer_num[0],growth_rate,self.feature_channel_num,middle_channels)
        self.feature_channel_num=self.feature_channel_num+layer_num[0]*growth_rate
        self.Transition1=Transition(self.feature_channel_num)

        self.DenseBlock2=DenseBlock(layer_num[1],growth_rate,self.feature_channel_num//2,middle_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[1]*growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2],growth_rate,self.feature_channel_num//2,middle_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[2]*growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3],growth_rate,self.feature_channel_num//2,middle_channels)
        self.feature_channel_num=self.feature_channel_num//2+layer_num[3]*growth_rate

        self.avgpool=nn.AdaptiveAvgPool1d(1)

        self.classifer = nn.Sequential(
            nn.Linear(self.feature_channel_num, self.feature_channel_num//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_channel_num//2, classes),
            nn.LogSoftmax(dim=1)
        )


    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.view(-1,self.feature_channel_num)
        x = self.classifer(x)

        return x
