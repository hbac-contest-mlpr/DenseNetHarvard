# https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch/blob/main/DenseNet.py
import torch
import torch.nn as nn
import torchinfo


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):

        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            self.add_module("denselayer%d" % (i), layer)


class Transition(nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm1d(channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module("Avgpool", nn.AvgPool1d(2))


class FeatureAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureAttentionModule, self).__init__()
        self.avg_pool = nn.AvgPool1d(3)
        self.max_pool = nn.MaxPool1d(3)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(self.avg_pool(x) + self.max_pool(x))
        y = self.sigmoid(y)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        layer_num=(5,),
        growth_rate=32,
        init_features=64,
        in_channels=1,
        classes=5,
    ):
        super(DenseNet, self).__init__()
        self.feature_channel_num = init_features
        self.conv = nn.Conv1d(in_channels, self.feature_channel_num, 7, 2, 3)
        self.norm = nn.BatchNorm1d(self.feature_channel_num)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.maxpool_adaptive = nn.AdaptiveMaxPool1d(1)

        self.DenseBlock1 = DenseBlock(
            layer_num[0], growth_rate, self.feature_channel_num
        )
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)
        
        self.FeatureAttentionModule1 = FeatureAttentionModule(self.feature_channel_num//2)
        self.feature_channel_num = self.feature_channel_num // 2
        self.ResidualBlock1 = ResidualBlock(
            self.feature_channel_num, self.feature_channel_num
        )
        
        self.DenseBlock2 = DenseBlock(
            layer_num[0], growth_rate, self.feature_channel_num
        )
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)
        
        self.FeatureAttentionModule2 = FeatureAttentionModule(self.feature_channel_num//2)
        self.feature_channel_num = self.feature_channel_num // 2
        self.ResidualBlock2 = ResidualBlock(
            self.feature_channel_num, self.feature_channel_num
        )
        
        self.DenseBlock3 = DenseBlock(
            layer_num[0], growth_rate, self.feature_channel_num
        )
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)
        
        self.FeatureAttentionModule3 = FeatureAttentionModule(self.feature_channel_num//2)
        self.feature_channel_num = self.feature_channel_num // 2
        self.ResidualBlock3 = ResidualBlock(
            self.feature_channel_num, self.feature_channel_num
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifer = nn.Sequential(
            nn.Linear(self.feature_channel_num, self.feature_channel_num // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_channel_num // 2, classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)
        x = self.FeatureAttentionModule1(x)
        x = self.ResidualBlock1(x)
        x = self.maxpool(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)
        x = self.FeatureAttentionModule2(x)
        x = self.ResidualBlock2(x)
        x = self.maxpool(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)
        x = self.FeatureAttentionModule3(x)
        x = self.ResidualBlock3(x)
        x = self.maxpool_adaptive(x) 

        x = x.view(-1, self.feature_channel_num)
        x = self.classifer(x)

        return x


if __name__ == "__main__":
    model = DenseNet(layer_num=(5,), growth_rate=32, in_channels=4, classes=6)  # model
    torchinfo.summary(model)
