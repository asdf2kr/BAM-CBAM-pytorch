import torch
import torch.nn as nn
from Models.conv import conv1x1, conv3x3, conv7x7

class BAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation):
        super(BAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        self.bn2_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = conv1x1(in_channel, self.hid_channel)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv2 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv3 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv4 = conv1x1(self.hid_channel, 1)
        self.bn4_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf

#To-do:
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation=1):
        super(CBAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv7x7(2, 1, stride=1, dilation=self.dilation)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc * x

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2
