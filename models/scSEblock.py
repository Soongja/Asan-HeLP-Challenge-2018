import torch
import torch.nn as nn


class SpatialGate3d(nn.Module):
    def __init__(self, channels):
        super(SpatialGate3d, self).__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sqz = self.conv(x)
        sqz = self.sigmoid(sqz)
        return sqz * x


class ChannelGate3d(nn.Module):
    def __init__(self, channels, reduction):
        super(ChannelGate3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sqz = self.avg_pool(x)
        sqz = self.fc1(sqz)
        sqz = self.relu(sqz)
        sqz = self.fc2(sqz)
        sqz = self.sigmoid(sqz)
        return sqz * x


class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate3d(channels)
        self.channel_gate = ChannelGate3d(channels, reduction=reduction)

    def  forward(self, x):
        sSE = self.spatial_gate(x)
        cSE = self.channel_gate(x)
        return sSE + cSE
