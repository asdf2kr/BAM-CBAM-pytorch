import torch
import torch.nn as nn
from Models.attention import BAM, CBAM
from Models.conv import conv1x1, conv3x3

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, hid_channels, atte='bam', ratio=16, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.downsample = downsample

        if atte == 'cbam':
            self.atte = CBAM(out_channels, ratio)
        else:
            self.atte = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # CBAM
        if not self.atte is None:
            out = self.atte(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module): # bottelneck-block, over the 50 layers.
    expansion = 4
    def __init__(self, in_channels, hid_channels, atte='bam', ratio=16, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        out_channels = hid_channels * self.expansion
        self.conv1 = conv1x1(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm2d(hid_channels)

        self.conv2 = conv3x3(hid_channels, hid_channels, stride)
        self.bn2 = nn.BatchNorm2d(hid_channels)

        self.conv3 = conv1x1(hid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if atte == 'cbam':
            self.atte = CBAM(out_channels, ratio)
        else:
            self.atte = None

    def forward(self, x):
        residual = x # indentity
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.atte is None:
            out = self.atte(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    '''
    *50-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 6
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 3.8x10^9
    '''
    '''
    *101-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 23
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 7.6x10^9
    '''
    def __init__(self, block, layers, num_classes=1000, atte='bam', ratio=16, dilation=4):
        super(ResNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.layers = layers
        self.in_channels = 64
        self.atte = atte
        self.ratio = ratio
        self.dilation = dilation

        if num_classes == 1000:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        if self.atte == 'bam':
            self.bam1 = BAM(64*block.expansion, self.ratio, self.dilation)
            self.bam2 = BAM(128*block.expansion, self.ratio, self.dilation)
            self.bam3 = BAM(256*block.expansion, self.ratio, self.dilation)

        self.conv2 = self.get_layers(block, 64, self.layers[0])
        self.conv3 = self.get_layers(block, 128, self.layers[1], stride=2)
        self.conv4 = self.get_layers(block, 256, self.layers[2], stride=2)
        self.conv5 = self.get_layers(block, 512, self.layers[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        torch.nn.init.kaiming_normal(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        torch.nn.init.kaiming_normal(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, hid_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.in_channels, hid_channels * block.expansion, stride),
                    nn.BatchNorm2d(hid_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, hid_channels, self.atte, self.ratio, stride, downsample))
        self.in_channels = hid_channels * block.expansion

        for _ in range(1, n_layers):
            layers.append(block(self.in_channels, hid_channels, self.atte, self.ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
            Example tensor shape based on resnet101
        '''

        x = self.conv1(x)

        x = self.conv2(x)
        if self.atte == 'bam':
            x = self.bam1(x)

        x = self.conv3(x)
        if self.atte == 'bam':
            x = self.bam2(x)

        x = self.conv4(x)
        if self.atte == 'bam':
            x = self.bam3(x)

        x = self.conv5(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    ''' ResNet-101 Model'''
    return ResNet(BottleneckBlock, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
