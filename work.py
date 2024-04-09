# using https://arxiv.org/pdf/1608.06993.pdf

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_size, kernel_size):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0
        self.num_bottleneck_output_filters = growth_rate * bottleneck_size
        if self.use_bottleneck:
            self.bn2 = nn.BatchNorm1d(in_channels)
            self.act2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(
                in_channels, 
                self.num_bottleneck_output_filters,
                kernel_size=1,
                stride=1)
        self.bn1 = nn.BatchNorm1d(self.num_bottleneck_output_filters)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            self.num_bottleneck_output_filters,
            growth_rate,
            kernel_size=kernel_size,
            stride=1, 
            dilation=1, 
            padding=kernel_size // 2)
        
    def forward(self, x):
        if self.use_bottleneck:
            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv2(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        return x

class DenseBlock(nn.ModuleDict):
    '''
    '''
    def __init__(self, num_layers, in_channels, growth_rate, kernel_size, bottleneck_size):
        super().__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.add_module(f'denselayer{i}', 
                DenseLayer(in_channels + i * growth_rate, 
                           growth_rate, 
                           bottleneck_size, 
                           kernel_size))

    def forward(self, x):
        layer_outputs = [x]
        for _, layer in self.items():
            x = layer(x)
            layer_outputs.append(x)
            x = torch.cat(layer_outputs, dim=1)
        return x
    

class TransitionBlock(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, dilation=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    

class DenseNet1d(nn.Module):

    def __init__(
        self, 
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        bottleneck_size: int = 4,
        kernel_size: int = 3, 
        in_channels: int = 3,
        num_classes: int = 1,
        reinit: bool = True,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
        nn.Conv1d(
            in_channels, num_init_features, 
            kernel_size=7, stride=2, padding=3, dilation=1),
        nn.BatchNorm1d(num_init_features),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
            )