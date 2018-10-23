import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, n=7):
        super(ResNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n, 1)
        self.layer2 = self._make_layer(16, 32, n, 2)
        self.layer3 = self._make_layer(32, 64, n, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, channels_in, channels_out, layer_count, stride):
        layers = []
        layers.append(ResidualBlock(channels_in, channels_out, stride))
        for _ in range(1, layer_count):
            layers.append(ResidualBlock(channels_out, channels_out))

        return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ResidualBlock, self).__init__()

        if channels_in != channels_out:
            self.downsample = ConvProjection(channels_in, channels_out, stride)
        else:
            self.downsample = None
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class ConvProjection(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 1, stride=stride)
    
    def forward(self, x):
        out = self.conv(x)
        return out
