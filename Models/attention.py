import torch
import torch.nn as nn
from Models.conv import conv1x1, conv3x3

class BAM(nn.Module):
    def __init__(self, in_channel, out_channel, reduction_ratio, dilation):
        super(BAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.out_channel = out_channel

        self.globalAvgPool = nn.AvgPool2d(self.out_channel, stride = 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        # self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        # self.bn2_1d = nn.BatchNorm1d(self.hid_channel)
        self.bn_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = conv1x1(in_channel, self.hid_channel)
        self.conv2 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.conv3 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.conv4 = conv1x1(self.hid_channel, 1)
        self.bn_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.relu(Mc)

        Mc = self.bn_1d(Mc)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.relu(Ms)

        Ms = self.bn_2d(Ms)
        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))

        Mf = 1 + self.sigmoid(Mc * Ms)
        return x + (x * Mf)

#To-do:
'''
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
    def forward(self, x):
        return x
'''
